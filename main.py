import gc
import json
import os
from typing import Any, Dict, Tuple

import torch
import yaml
from torch import multiprocessing as mp

from agi.brain import brain_proc
from agi.commander import commander_proc
from agi.ctx import ctx_proc
from agi.governor import governor_proc
from agi.memorizer import memory_manager_proc
from agi.motivator import motivator_proc
from agi.perceive import perceive_proc
from src.utilities.mj.mjcf import get_humanoid_geoms
from src.utilities.tools.retry import retry


class RuntimeEngine:
    def __init__(self):
        self.shared_guidances: Dict[str, mp.Queue] = {}
        self.shared_memory: Dict[str, torch.Tensor] = {}
        self.metadata: Dict[str, Any] = {}

    def add_guidance(self, name: str, guidance: mp.Queue):
        self.shared_guidances[name] = guidance

    def get_guidance(self, name: str) -> mp.Queue:
        return self.shared_guidances[name]

    def add_shm(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype):
        shm = torch.zeros(shape, dtype=dtype)
        self.shared_memory[name] = shm
        shm.share_memory_()

    def update_shm(
        self,
        name: str,
        tensor: torch.Tensor,
        slice_: slice | None = None,
    ) -> None:
        if torch.any(torch.isnan(tensor)):
            print(f"writing nan to {name}. skipping.")
            return None

        if slice_ is None:
            self.shared_memory[name].copy_(tensor, non_blocking=True)
        else:
            self.shared_memory[name][slice_] = tensor

    def get_shm(self, name: str) -> torch.Tensor | None:
        if torch.any(torch.isnan(self.shared_memory[name])):
            print(f"reading nan from {name}. skipping.")
            return None
        return self.shared_memory[name]

    def add_metadata(self, name: str, metadata: Any):
        self.metadata[name] = metadata

    def get_metadata(self, name: str) -> Any:
        return self.metadata[name]


class Host:
    def __init__(
        self,
        cfg_file,
        exp_dir,
    ):
        runtime_engine = RuntimeEngine()
        self.cfg_file = cfg_file
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = yaml.safe_load(open(self.cfg_file))
        self.cfg = cfg
        intrinsics = cfg["intrinsics"]
        intrinsic_indices = {intrinsics[i]: i for i in range(len(intrinsics))}
        robot_props = self.connect_robot()

        runtime_engine.add_metadata("robot_props", robot_props)
        runtime_engine.add_metadata("config", cfg)
        runtime_engine.add_metadata("device", device)
        runtime_engine.add_metadata("intrinsics", intrinsics)
        runtime_engine.add_metadata("intrinsic_indices", intrinsic_indices)

        # shm
        latent_offset = 0
        latent_slices = {}

        for name, params in cfg["perceivers"].items():
            assert params["emb_dim"] > 0, f"emb_dim must be greater than 0 in {name}"
            emb_dim = params["emb_dim"]
            latent_slices[name] = slice(latent_offset, latent_offset + emb_dim)
            latent_offset += emb_dim

        runtime_engine.add_metadata("latent_slices", latent_slices)

        # queues
        torque_guidance_q = mp.Queue()
        emotion_guidance_q = mp.Queue()
        runtime_engine.add_guidance("torque", torque_guidance_q)
        runtime_engine.add_guidance("emotion", emotion_guidance_q)

        runtime_engine.add_shm(
            "human_state",
            (10,),
            torch.float64,
        )

        runtime_engine.add_shm(
            "vision",
            (
                3,
                robot_props["egocentric_view_height"],
                robot_props["egocentric_view_width"],
            ),
            torch.float32,
        )
        runtime_engine.add_shm(
            "qpos",
            (robot_props["nq"],),
            torch.float32,
        )
        runtime_engine.add_shm(
            "qvel",
            (robot_props["nv"],),
            torch.float32,
        )
        runtime_engine.add_shm(
            "force_on_geoms",
            (robot_props["n_geoms"],),
            torch.float32,
        )

        runtime_engine.add_shm(
            "latent",
            (1, latent_offset),
            torch.float32,
        )
        runtime_engine.add_shm(
            "emotions",
            (1, cfg["emotion"]["pad_dim"]),
            torch.float32,
        )

        runtime_engine.add_shm(
            "torques",
            (1, robot_props["n_actuators"]),
            torch.float32,
        )

        runtime_engine.add_shm(
            "governance",
            (len(intrinsics),),
            torch.float32,
        )

        ctx_proc0 = mp.Process(
            target=ctx_proc,
            args=(runtime_engine,),
        )

        perceive_proc0 = mp.Process(
            target=perceive_proc,
            args=(
                runtime_engine,
                "vision",
            ),
        )

        memory_manager_proc0 = mp.Process(
            target=memory_manager_proc,
            args=(
                runtime_engine,
                "live_memory",
            ),
        )

        memory_manager_proc1 = mp.Process(
            target=memory_manager_proc,
            args=(
                runtime_engine,
                "episodic_memory",
            ),
        )

        governor_proc0 = mp.Process(
            target=governor_proc,
            args=(runtime_engine,),
        )

        motivator_proc0 = mp.Process(
            target=motivator_proc,
            args=(runtime_engine,),
        )

        brain_proc0 = mp.Process(
            target=brain_proc,
            args=(runtime_engine,),
        )

        commander_proc0 = mp.Process(
            target=commander_proc,
            args=(runtime_engine,),
        )

        ctx_proc0.start()
        perceive_proc0.start()
        memory_manager_proc0.start()
        memory_manager_proc1.start()
        motivator_proc0.start()
        brain_proc0.start()
        governor_proc0.start()
        commander_proc0.start()

        ctx_proc0.join()
        perceive_proc0.join()
        memory_manager_proc0.join()
        memory_manager_proc1.join()
        brain_proc0.join()
        motivator_proc0.join()
        governor_proc0.join()
        commander_proc0.join()

    @retry
    def connect_robot(self) -> Dict[str, Any]:
        import zmq
        from agi.ctx import CTXParser
        from dotenv import load_dotenv

        load_dotenv()

        robot_cfg = self.cfg["robot"]
        print("connecting to robot.")
        zmq_tmp_ctx = zmq.Context()
        sub = zmq_tmp_ctx.socket(zmq.SUB)
        robot_sub_cfg = robot_cfg["sub"]
        ip = os.getenv("WINDOWS_IP", robot_sub_cfg["ip"])
        print("robot_sub_cfg: ", ip)
        url = f"{robot_sub_cfg['protocol']}://{ip}:{robot_sub_cfg['port']}"  # type: ignore
        print(url)
        sub.connect(url)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        frame: dict = sub.recv_json()  # type: ignore
        sub.close()
        zmq_tmp_ctx.term()
        print("robot connected.")

        ctx_parser = CTXParser(robot_cfg)
        humanoid_geoms = get_humanoid_geoms(robot_cfg["mjcf_path"])
        robot_state = json.loads(frame["robot_state"])
        robot_state["egocentric_view"] = bytes(frame["egocentric_view"])
        geom_mapping = robot_state["geom_mapping"]["geom_name_id_mapping"]
        humanoid_geom_mapping = {
            k: v
            for k, v in geom_mapping.items()
            if any(k.startswith(humanoid_geom) for humanoid_geom in humanoid_geoms)
        }
        humanoid_geom_mapping_rev = {v: k for k, v in humanoid_geom_mapping.items()}
        humanoid_indices = [
            v
            for k, v in geom_mapping.items()
            if any(k.startswith(humanoid_geom) for humanoid_geom in humanoid_geoms)
        ]
        joint_mapping = robot_state["joint_mapping"]["joint_name_id_mapping"]
        geom_mapping_rev = {v: k for k, v in geom_mapping.items()}
        joint_mapping_rev = {v: k for k, v in joint_mapping.items()}

        actuator_mapping = robot_state["actuator_mapping"]["actuator_name_id_mapping"]
        actuator_mapping_rev = {v: k for k, v in actuator_mapping.items()}

        egocentric_view = ctx_parser.parse(robot_state, "vision")
        if egocentric_view is None:
            raise ValueError("egocentric_view is None")

        robot_props = {
            "geom_id2name": geom_mapping_rev,
            "geom_name2id": geom_mapping,
            "humanoid_geom_name2id": humanoid_geom_mapping,
            "humanoid_geom_id2name": humanoid_geom_mapping_rev,
            "n_geoms": len(geom_mapping),
            "n_body_geoms": len(geom_mapping),
            "n_humanoid_geoms": len(humanoid_geom_mapping),
            "n_actuators": len(actuator_mapping),
            "humanoid_geom_indices": humanoid_indices,
            "joint_id2name": joint_mapping_rev,
            "joint_name2id": joint_mapping,
            "actuator_id2name": actuator_mapping_rev,
            "actuator_name2id": actuator_mapping,
            "egocentric_view_width": egocentric_view.shape[2],
            "egocentric_view_height": egocentric_view.shape[1],
            "nq": len(robot_state["qpos"]),
            "nv": len(robot_state["qvel"]),
        }

        return robot_props


if __name__ == "__main__":
    import platform
    import subprocess
    from argparse import ArgumentParser
    from src.utilities.docker.container import running_containers
    from src.utilities.nvidia.nvidia_smi import get_nvidia_process_names

    mp.set_start_method("spawn", force=True)
    print("available start methods:", mp.get_all_start_methods())
    print(f"available CPU cores: {mp.cpu_count()}")

    if platform.system() == "Linux":
        import shutil

        if "qdrant" not in running_containers():
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-d",
                    "--name",
                    "qdrant",
                    "-p",
                    "6333:6333",
                    "-v",
                    f"{os.getcwd()}/qdrant_storage:/qdrant/storage",
                    "qdrant/qdrant",
                ]
            )

        if (
            shutil.which("nvidia-cuda-mps-control")
            and "nvidia-cuda-mps-server" not in get_nvidia_process_names()
        ):
            print("starting mps")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
            subprocess.run(["nvidia-cuda-mps-control", "-d"])

    elif platform.system() == "Windows":
        if "qdrant" not in running_containers():
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-d",
                    "--name",
                    "qdrant",
                    "-p",
                    "6333:6333",
                    "-v",
                    f"{os.getcwd()}\\qdrant_storage:/qdrant/storage",
                    "qdrant/qdrant",
                ]
            )

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.template.yaml")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # TODO: fix it.
    os.environ["DEBUG"] = str(args.debug)

    host = Host(
        cfg_file=args.config,
        exp_dir=args.exp_dir,
    )
    gc.collect()
    torch.cuda.empty_cache()

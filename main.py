import gc
import json
import os
from typing import Any, Dict

import yaml
from loguru import logger
import torch
from torch import multiprocessing as mp

from agi import (
    ctx_parser,
    perceiver,
    memory_manager,
    governor,
    motivator,
    brain,
    actuator,
)
from src.utilities.mj.mjcf import get_humanoid_geoms
from src.utilities.tools.retry import retry


class RuntimeEngine:
    def __init__(self):
        self.shared_memory: Dict[str, Dict[str, mp.Queue]] = {}
        self.metadata: Dict[str, Any] = {}

    def add_shared_memory(self, name: str, shared_memory: Dict[str, mp.Queue]):
        """
        stores shared memory for agi
        """
        self.shared_memory[name] = shared_memory

    def get_shared_memory(self, name: str) -> Dict[str, mp.Queue]:
        return self.shared_memory[name]

    def add_metadata(self, name: str, metadata: Any):
        """
        stores agi metadata
        """
        self.metadata[name] = metadata

    def get_metadata(self, name: str) -> Any:
        return self.metadata[name]


class Host:
    def __init__(
        self,
        cfg_file,
        exp_dir,
    ):
        # ----------configuration----------
        runtime_engine = RuntimeEngine()
        self.cfg_file = cfg_file
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = yaml.safe_load(open(self.cfg_file))
        if os.environ.get("RUNNING_ENV") == "docker":
            cfg["robot"]["sub"]["ip"] = "host.docker.internal"
            cfg["robot"]["pub"]["ip"] = "host.docker.internal"
            cfg["robot"]["mjcf_path"] = "/app/simulator/Assets/MJCF/humanoid.xml"
        self.cfg = cfg
        intrinsics = cfg["intrinsics"]
        intrinsic_indices = {intrinsics[i]: i for i in range(len(intrinsics))}
        robot_props = self.connect_robot()

        # ----------metadata----------
        latent_offset = 0
        latent_slices = {}

        for name, params in cfg["perceivers"].items():
            assert params["emb_dim"] > 0, f"emb_dim must be greater than 0 in {name}"
            emb_dim = params["emb_dim"]
            latent_slices[name] = slice(latent_offset, latent_offset + emb_dim)
            latent_offset += emb_dim

        runtime_engine.add_metadata("robot_props", robot_props)
        runtime_engine.add_metadata("config", cfg)
        runtime_engine.add_metadata("device", device)
        runtime_engine.add_metadata("intrinsics", intrinsics)
        runtime_engine.add_metadata("intrinsic_indices", intrinsic_indices)
        runtime_engine.add_metadata("latent_slices", latent_slices)
        runtime_engine.add_metadata("latent_size", latent_offset)

        # ----------shared memory----------
        perceiver_shm = {
            "vision": mp.Queue(),
        }
        memory_manager_shm = {
            "vision_latent": mp.Queue(),
            "emotion": mp.Queue(),
            "torque": mp.Queue(),
        }
        motivator_shm = {
            "emotion": mp.Queue(),
            "governance": mp.Queue(),
            "latent": mp.Queue(),
            "robot_state": mp.Queue(),
            "force_on_geoms": mp.Queue(),
        }
        governor_shm = {
            "emotion": mp.Queue(),
        }
        brain_shm = {
            "latent": mp.Queue(),
            "emotion": mp.Queue(),
            "torque": mp.Queue(),
        }
        actuator_shm = {
            "torque": mp.Queue(),
        }

        runtime_engine.add_shared_memory("perceiver", perceiver_shm)
        runtime_engine.add_shared_memory("memory_manager", memory_manager_shm)
        runtime_engine.add_shared_memory("brain", brain_shm)
        runtime_engine.add_shared_memory("motivator", motivator_shm)
        runtime_engine.add_shared_memory("governor", governor_shm)
        runtime_engine.add_shared_memory("actuator", actuator_shm)

        # ---------process----------
        ctx_parser_process = mp.Process(target=ctx_parser, args=(runtime_engine,))
        perceiver_vision_process = mp.Process(
            target=perceiver, args=(runtime_engine, "vision")
        )
        memory_manager_live_process = mp.Process(
            target=memory_manager, args=(runtime_engine, "live_memory")
        )
        memory_manager_episodic_process = mp.Process(
            target=memory_manager, args=(runtime_engine, "episodic_memory")
        )
        governor_process = mp.Process(target=governor, args=(runtime_engine,))
        motivator_process = mp.Process(target=motivator, args=(runtime_engine,))
        brain_process = mp.Process(target=brain, args=(runtime_engine,))
        actuator_process = mp.Process(target=actuator, args=(runtime_engine,))

        ctx_parser_process.start()
        perceiver_vision_process.start()
        memory_manager_live_process.start()
        memory_manager_episodic_process.start()
        governor_process.start()
        motivator_process.start()
        brain_process.start()
        actuator_process.start()

        ctx_parser_process.join()
        perceiver_vision_process.join()
        memory_manager_live_process.join()
        memory_manager_episodic_process.join()
        governor_process.join()
        motivator_process.join()
        brain_process.join()
        actuator_process.join()

    @retry
    def connect_robot(self) -> Dict[str, Any]:
        import zmq
        from dotenv import load_dotenv
        from PIL import Image
        import io
        import torchvision.transforms as transforms

        load_dotenv()

        robot_cfg = self.cfg["robot"]
        logger.info("connecting to robot.")
        zmq_tmp_ctx = zmq.Context()
        sub = zmq_tmp_ctx.socket(zmq.SUB)
        robot_sub_cfg = robot_cfg["sub"]
        ip = os.getenv("WINDOWS_IP", robot_sub_cfg["ip"])
        logger.info("robot_sub_cfg: ", ip)
        url = f"{robot_sub_cfg['protocol']}://{ip}:{robot_sub_cfg['port']}"  # type: ignore
        logger.info(url)
        sub.connect(url)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        frame: dict = sub.recv_json()  # type: ignore
        sub.close()
        zmq_tmp_ctx.term()
        logger.info("robot connected.")
        logger.info(frame.keys())

        humanoid_geoms = get_humanoid_geoms(robot_cfg["mjcf_path"])
        robot_state = json.loads(frame["robot_state"])
        robot_mapping = json.loads(frame["robot_mapping"])
        robot_state["egocentric_view"] = bytes(frame["egocentric_view"])
        geom_mapping = robot_mapping["geom_name_id_mapping"]
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
        joint_mapping = robot_mapping["joint_name_id_mapping"]
        geom_mapping_rev = {v: k for k, v in geom_mapping.items()}
        joint_mapping_rev = {v: k for k, v in joint_mapping.items()}

        actuator_mapping = robot_mapping["actuator_name_id_mapping"]
        actuator_mapping_rev = {v: k for k, v in actuator_mapping.items()}

        img_data = bytes(robot_state["egocentric_view"])
        img = Image.open(io.BytesIO(img_data))
        transform = transforms.ToTensor()
        img = transform(img)
        if torch.any(torch.isnan(img)):
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
            "egocentric_view_width": img.shape[2],
            "egocentric_view_height": img.shape[1],
            "n_qpos": len(robot_state["qpos"]),
            "n_qvel": len(robot_state["qvel"]),
        }

        return robot_props


if __name__ == "__main__":
    import platform
    import subprocess
    from argparse import ArgumentParser
    from src.utilities.docker.container import running_containers

    mp.set_start_method("spawn", force=True)
    running_env = os.environ.get("RUNNING_ENV")
    logger.info("available start methods:", mp.get_all_start_methods())
    logger.info(f"available CPU cores: {mp.cpu_count()}")

    if platform.system() == "Linux":
        if running_env != "docker" and "qdrant" not in running_containers():
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
    elif platform.system() == "Windows":
        if running_env != "docker" and "qdrant" not in running_containers():
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

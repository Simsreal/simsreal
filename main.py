import gc
import json
import os

import torch
import yaml
from torch import multiprocessing as mp

from human.process import (
    brain_proc,
    commander_proc,
    ctx_proc,
    governor_proc,
    memory_manager_proc,
    motivator_proc,
    perceive_proc,
)
from utilities.mj.mjcf import get_humanoid_geoms

# class SHMManager:


class Host:
    def __init__(
        self,
        cfg_file,
        exp_dir,
    ):
        self.cfg_file = cfg_file
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = yaml.safe_load(open(self.cfg_file))
        self.cfg = cfg
        robot_cfg = cfg["robot"]
        perceivers_cfg = cfg["perceivers"]
        intrinsics = cfg["intrinsics"]
        intrinsic_indices = {intrinsics[i]: i for i in range(len(intrinsics))}

        # queues
        drives_q = mp.Queue()
        emotions_q = mp.Queue()

        queues = {
            "drives_q": drives_q,
            "emotions_q": emotions_q,
        }

        robot_info = self.initialize_robot_info(robot_cfg)
        # shm
        latent_offset = 0
        latent_slices = {}

        for name, params in perceivers_cfg.items():
            emb_dim = params.get("emb_dim", 0)
            latent_slices[name] = slice(latent_offset, latent_offset + emb_dim)
            latent_offset += emb_dim

        human_state = torch.zeros(
            10,  # max. number of human states
            dtype=torch.float64,
        )

        vision = torch.zeros(
            (
                3,
                robot_info["egocentric_view_height"],
                robot_info["egocentric_view_width"],
            ),
            dtype=torch.float32,
        )

        qpos = torch.zeros(
            robot_info["nq"],
            dtype=torch.float32,
        )

        qvel = torch.zeros(
            robot_info["nv"],
            dtype=torch.float32,
        )

        governance = torch.zeros(
            (len(intrinsics),),
            dtype=torch.float32,
        )

        latent = torch.zeros(
            (
                1,
                latent_offset,
            ),
            dtype=torch.float32,
        )

        torques = torch.zeros((1, robot_info["n_actuators"]), dtype=torch.float32)

        emotions = torch.zeros((1, cfg["emotion"]["pad_dim"]), dtype=torch.float32)

        force_on_geoms = torch.zeros(robot_info["n_geoms"], dtype=torch.float32)

        force_on_geoms.share_memory_()
        human_state.share_memory_()
        vision.share_memory_()
        qpos.share_memory_()
        qvel.share_memory_()
        governance.share_memory_()
        latent.share_memory_()
        torques.share_memory_()
        emotions.share_memory_()

        shm = {
            "human_state": human_state,
            "vision": vision,
            "qpos": qpos,
            "qvel": qvel,
            "force_on_geoms": force_on_geoms,
            "governance": governance,
            "latent": latent,
            "torques": torques,
            "emotions": emotions,
            "intrinsic_indices": intrinsic_indices,
            "robot_info": robot_info,
            "latent_slices": latent_slices,
            "device": device,
        }

        # proc
        ctx_proc0 = mp.Process(
            target=ctx_proc,
            args=(shm, cfg),
        )

        governor_proc0 = mp.Process(
            target=governor_proc,
            args=(
                shm,
                cfg,
            ),
        )

        perceive_proc0 = mp.Process(
            target=perceive_proc,
            args=(
                shm,
                "vision",
                perceivers_cfg,
            ),
        )

        memory_manager_proc0 = mp.Process(
            target=memory_manager_proc,
            args=(
                shm,
                "live_memory",
                cfg,
            ),
        )

        memory_manager_proc1 = mp.Process(
            target=memory_manager_proc,
            args=(
                shm,
                "episodic_memory",
                cfg,
            ),
        )

        motivator_proc0 = mp.Process(
            target=motivator_proc,
            args=(
                shm,
                queues,
                cfg,
            ),
        )

        brain_proc0 = mp.Process(
            target=brain_proc,
            args=(
                shm,
                queues,
                cfg,
            ),
        )

        commander_proc0 = mp.Process(
            target=commander_proc,
            args=(
                shm,
                cfg,
            ),
        )

        ctx_proc0.start()
        perceive_proc0.start()
        memory_manager_proc0.start()
        memory_manager_proc1.start()
        motivator_proc0.start()
        brain_proc0.start()
        governor_proc0.start()
        commander_proc0.start()

        brain_proc0.join()
        ctx_proc0.join()
        perceive_proc0.join()
        memory_manager_proc0.join()
        memory_manager_proc1.join()
        motivator_proc0.join()
        governor_proc0.join()
        commander_proc0.join()

    def initialize_robot_info(self, robot_cfg):
        import zmq

        from human.process.ctx import vision_parser

        print("connecting to robot.")
        zmq_tmp_ctx = zmq.Context()
        sub = zmq_tmp_ctx.socket(zmq.SUB)
        robot_sub_cfg = robot_cfg["sub"]
        url = f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"
        print(url)
        sub.connect(url)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        frame: dict = sub.recv_json()  # type: ignore
        sub.close()
        zmq_tmp_ctx.term()

        humanoid_geoms = get_humanoid_geoms(robot_cfg["mjcf_path"])
        robot_state = json.loads(frame["robot_state"])
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

        egocentric_view = vision_parser(frame)

        robot_info = {
            "geom_id2name": geom_mapping_rev,
            "geom_name2id": geom_mapping,
            "humanoid_geom_name2id": humanoid_geom_mapping,
            "humanoid_geom_id2name": humanoid_geom_mapping_rev,
            "n_geoms": len(geom_mapping),
            "n_body_geoms": len(geom_mapping),
            "n_humanoid_geoms": len(humanoid_geom_mapping),
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

        return robot_info


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("available start methods:", mp.get_all_start_methods())
    print(f"available CPU cores: {mp.cpu_count()}")
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("-uc", "--unconsciousness", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--silent", action="store_true")

    args = parser.parse_args()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["UNCONSCIOUS"] = str(args.unconsciousness)
    os.environ["VERBOSE"] = "silent" if args.silent else "verbose"
    os.environ["DEBUG"] = str(args.debug)

    host = Host(
        cfg_file=args.config,
        exp_dir=args.exp_dir,
    )
    gc.collect()
    torch.cuda.empty_cache()

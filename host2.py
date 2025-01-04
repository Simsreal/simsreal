import asyncio
import gc
import os
import platform
import traceback

import torch
import yaml
from tensordict import TensorDict
from torch import multiprocessing as mp

from human.process.brain_wrapper import brain_proc
from human.process.ctx_wrapper import ctx_proc
from human.process.neural_gate import neural_gate_proc
from human.process.perceive_wrapper import eye_proc

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class Hostv2:
    def __init__(
        self,
        cfg_file,
        exp_dir,
    ):
        self.cfg_file = cfg_file
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = yaml.safe_load(open(self.cfg_file))
        ctx_cfg = cfg["ctx"]
        intrinsics_cfg = cfg["intrinsics"]
        perceivers_cfg = cfg["perceivers"]

        vision_perceptor_cfg = perceivers_cfg["vision"]
        brain_cfg = cfg["brain"]

        # shm
        robot_info = self.initialize_robot_info()
        vision_tensor = torch.zeros(
            (
                ctx_cfg["vision"]["h"],
                ctx_cfg["vision"]["w"],
                3,
            ),
            dtype=torch.float32,
        )

        qpos_tensor = torch.zeros(
            ctx_cfg["qpos"]["dim"],
            dtype=torch.float32,
        )

        qvel_tensor = torch.zeros(
            ctx_cfg["qvel"]["dim"],
            dtype=torch.float32,
        )

        contact_tensor = torch.zeros(
            (len(robot_info["n_body_geoms"]), ctx_cfg["contact"]["dim"])
        )
        print(contact_tensor)
        exit()

        neural_gate = torch.zeros(
            len(intrinsics_cfg),
            dtype=torch.float32,
        )

        latent = torch.zeros(
            (
                1,
                vision_perceptor_cfg["emb_dim"],
            ),
            dtype=torch.float32,
        )

        latent.share_memory_()
        vision_tensor.share_memory_()
        qpos_tensor.share_memory_()
        qvel_tensor.share_memory_()
        neural_gate.share_memory_()

        shm = {
            "vision": vision_tensor,
            "qpos": qpos_tensor,
            "qvel": qvel_tensor,
            "neural_gate": neural_gate,
            "latent": latent,
            "robot_info": robot_info,
        }

        # slice
        brain_slices = {
            "vision": slice(0, vision_perceptor_cfg["emb_dim"]),
        }

        # proc
        ctx_proc0 = mp.Process(
            target=ctx_proc,
            args=(cfg, shm),
        )

        neural_gate0 = mp.Process(
            target=neural_gate_proc,
            args=(
                cfg,
                shm,
            ),
        )

        perceive_proc0 = mp.Process(
            target=eye_proc,
            args=(
                shm,
                brain_slices,
                self.device,
            ),
        )

        brain_proc0 = mp.Process(
            target=brain_proc,
            args=(
                shm,
                brain_cfg,
                self.device,
            ),
        )

        self.wrappers = [
            ctx_proc0,
            perceive_proc0,
            brain_proc0,
            neural_gate0,
        ]

    def initialize_robot_info(
        self,
        max_string_len=100,
    ):
        def string_to_tensor(x):
            return torch.tensor(
                [ord(c) for c in x],
                dtype=torch.uint8,
            )

        import zmq

        zmq_tmp_ctx = zmq.Context()
        sub = zmq_tmp_ctx.socket(zmq.SUB)
        sub.connect("tcp://127.0.0.1:5556")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        msg = sub.recv_pyobj()["aji6"]
        sub.close()
        zmq_tmp_ctx.term()

        robot_geoms = msg["body_geoms"]
        # geoms_id2name = msg["geom_mapping"]["geom_id_to_name"]
        geoms_name2id = msg["geom_mapping"]["geom_name_to_id"]

        geom_id2name = TensorDict(
            **{
                str(geoms_name2id[geom_name]): string_to_tensor(geom_name)
                for geom_name in robot_geoms
            }
        )
        geom_name2id = TensorDict(
            **{
                geom_name: torch.tensor([geoms_name2id[geom_name]], dtype=torch.uint8)
                for geom_name in robot_geoms
            }
        )
        return {
            "geom_id2name": geom_id2name,
            "geom_name2id": geom_name2id,
            "n_body_geoms": len(robot_geoms),
        }

    def run(self):
        print("starting")
        for wrapper in self.wrappers:
            wrapper.start()

    def stop(self):
        for wrapper in self.wrappers:
            wrapper.join()

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("available start methods:", mp.get_all_start_methods())
    print(f"available CPU cores: {mp.cpu_count()}")
    mp.set_start_method("spawn", force=True)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="human_config/aji6.yaml")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("-uc", "--unconsciousness", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--silent", action="store_true")

    args = parser.parse_args()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
    os.environ["UNCONSCIOUS"] = str(args.unconsciousness)
    os.environ["VERBOSE"] = "silent" if args.silent else "verbose"
    os.environ["DEBUG"] = str(args.debug)

    host = Hostv2(
        cfg_file=args.config,
        exp_dir=args.exp_dir,
    )
    try:
        host.run()
    except KeyboardInterrupt:
        host.stop()
    except Exception:
        traceback.print_exc()
        host.stop()

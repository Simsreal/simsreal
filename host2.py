import gc
import os
import traceback

import torch
import yaml
from torch import multiprocessing as mp

from human.memory.brain.lstm import LSTM
from human.memory.perceive.retina import Retina
from human.process.brain import brain_proc
from human.process.commander import commander_proc
from human.process.ctx import ctx_proc
from human.process.neural_gate import neural_gate_proc
from human.process.perceive import perceive_proc


class Hostv2:
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
        ctx_cfg = cfg["ctx"]
        perceivers_cfg = cfg["perceivers"]
        intrinsics = cfg["intrinsics"]
        emotion_cfg = cfg["emotion"]
        gate_indices = {intrinsics[i]: i for i in range(len(intrinsics))}

        vision_perceptor_cfg = perceivers_cfg["vision"]
        brain_cfg = cfg["brain"]

        # queues
        # drives_q = mp.Queue()

        # queues = {
        #     "drives_q": drives_q,
        # }

        # shm
        robot_info = self.initialize_robot_info(robot_cfg)
        brain_slices = {
            "vision": slice(0, vision_perceptor_cfg["emb_dim"]),
        }

        vision = torch.zeros(
            (
                ctx_cfg["vision"]["h"],
                ctx_cfg["vision"]["w"],
                3,
            ),
            dtype=torch.float32,
        )

        qpos = torch.zeros(
            ctx_cfg["qpos"]["dim"],
            dtype=torch.float32,
        )

        qvel = torch.zeros(
            ctx_cfg["qvel"]["dim"],
            dtype=torch.float32,
        )

        contact = torch.zeros(
            (robot_info["n_geoms"], ctx_cfg["contact"]["dim"]), dtype=torch.float32
        )

        neural_gate = torch.zeros(
            (len(intrinsics),),
            dtype=torch.float32,
        )

        latent = torch.zeros(
            (
                1,
                vision_perceptor_cfg["emb_dim"],
            ),
            dtype=torch.float32,
        )

        torques = torch.zeros((1, brain_cfg["n_actuators"]), dtype=torch.float32)

        emotions = torch.zeros((1, emotion_cfg["pad_dim"]), dtype=torch.float32)

        vision.share_memory_()
        contact.share_memory_()
        qpos.share_memory_()
        qvel.share_memory_()
        neural_gate.share_memory_()
        latent.share_memory_()
        torques.share_memory_()
        emotions.share_memory_()

        shm = {
            "vision": vision,
            "qpos": qpos,
            "qvel": qvel,
            "contact": contact,
            "neural_gate": neural_gate,
            "latent": latent,
            "torques": torques,
            "emotions": emotions,
            "gate_indices": gate_indices,
            "robot_info": robot_info,
            "brain_slices": brain_slices,
            "device": device,
        }

        # memory
        brain_cfg["latent_size"] = latent.shape[-1]
        retina = Retina(emb_dim=perceivers_cfg["vision"]["emb_dim"]).to(device)
        lstm = LSTM(
            brain_cfg["latent_size"],
            brain_cfg["hidden_dim"],
            brain_cfg["n_layers"],
            device,
            1,
            brain_cfg["n_actuators"],
        ).to(device)

        retina.share_memory()
        lstm.share_memory()

        memory = {"retina": retina, "lstm": lstm}

        # proc
        ctx_proc0 = mp.Process(
            target=ctx_proc,
            args=(shm, cfg),
        )

        neural_gate0 = mp.Process(
            target=neural_gate_proc,
            args=(
                shm,
                cfg,
            ),
        )

        perceive_proc0 = mp.Process(
            target=perceive_proc,
            args=(
                shm,
                memory,
            ),
        )

        brain_proc0 = mp.Process(
            target=brain_proc,
            args=(
                shm,
                memory,
                brain_cfg,
            ),
        )

        commander_proc0 = mp.Process(
            target=commander_proc,
            args=(
                shm,
                cfg,
            ),
        )

        self.processes = [
            ctx_proc0,
            perceive_proc0,
            brain_proc0,
            neural_gate0,
            commander_proc0,
        ]

    def initialize_robot_info(self, robot_cfg):
        import zmq

        print("connecting to robot.")
        zmq_tmp_ctx = zmq.Context()
        sub = zmq_tmp_ctx.socket(zmq.SUB)
        robot_sub_cfg = robot_cfg["sub"]
        sub.connect(
            f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"
        )
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        msg = sub.recv_pyobj()[self.cfg["name"]]
        sub.close()
        zmq_tmp_ctx.term()

        robot_geoms = msg["body_geoms"]
        geoms_id2name = msg["geom_mapping"]["geom_id_to_name"]
        geoms_name2id = msg["geom_mapping"]["geom_name_to_id"]
        n_geoms = len(geoms_name2id)
        n_body_geoms = len(robot_geoms)

        return {
            "geom_id2name": geoms_id2name,
            "geom_name2id": geoms_name2id,
            "n_geoms": n_geoms,
            "n_body_geoms": n_body_geoms,
        }

    def run(self):
        print("starting")
        for wrapper in self.processes:
            wrapper.start()

    def stop(self):
        for wrapper in self.processes:
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

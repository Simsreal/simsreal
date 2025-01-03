import asyncio
import gc
import os
import platform
import traceback

import torch
import yaml
from torch import multiprocessing as mp

from human.wrappers import ContextWrapper
from human.wrappers.brain_wrapper import brain_proc
from human.wrappers.perceive_wrapper import eye_proc

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

        cfg = yaml.safe_load(open(self.cfg_file))
        ctx_cfg = cfg["ctx"]
        perceivers_cfg = cfg["perceivers"]

        vision_ctx_cfg = ctx_cfg["vision"]
        vision_perceptor_cfg = perceivers_cfg["vision"]

        # shm
        vision_tensor = torch.zeros(
            (
                vision_ctx_cfg["height"],
                vision_ctx_cfg["width"],
                3,
            ),
            dtype=torch.float32,
        )
        vision_tensor.share_memory_()

        # slice
        brain_slices = {
            "vision": slice(0, vision_perceptor_cfg["emb_dim"]),
        }
        aggregated_latent = torch.zeros(
            (
                1,
                vision_perceptor_cfg["emb_dim"],
            ),
            dtype=torch.float32,
        )

        # proc
        brain_cfg = cfg["brain"]

        self.ctx_wrapper: mp.Process = ContextWrapper(
            cfg=cfg,
            vision_tensor=vision_tensor,
        )

        perceive_proc0 = mp.Process(
            target=eye_proc,
            args=(
                vision_tensor,
                brain_slices,
                aggregated_latent,
            ),
        )

        brain_proc0 = mp.Process(
            target=brain_proc,
            args=(
                aggregated_latent,
                brain_cfg["ctx_len"],
            ),
        )
        self.wrappers = [
            self.ctx_wrapper,
            perceive_proc0,
            brain_proc0,
        ]

    def run(self):
        for wrapper in self.wrappers:
            wrapper.start()

    def stop(self):
        for wrapper in self.wrappers:
            wrapper.join()

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("available start methods:", mp.get_all_start_methods())
    mp.set_start_method("spawn", force=True)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="human_config/aji5.yaml")
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

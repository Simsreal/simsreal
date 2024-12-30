# flake8: noqa: F403, E741, F405
import gc
import os
import traceback
from typing import Any, Dict, List, Tuple

import httpx
import torch
import yaml

# from icecream import ic
from prometheus_client import start_http_server

from environment import *
from human.constraints import *
from human.context import *
from human.humans import *
from human.instincts import *
from human.memory.cerebrum import LSTM, xLSTMCerebrum
from human.memory.emotions import MCTS, PolicyValueNet
from human.neuro_symbol import *
from human.neuro_symbol.downward_planner import DownwardPlanner
from human.neuro_symbol.receipes import *
from human.perceptors import *
from intelligence.memory import Memory

# from simulators import *

CONFIG_DIR = "human_config"
DOWNWARD_PATH = "downward/fast-downward.py"
EXPERIMENT_DIR = "experiments"


class Host:
    Env: Dict[str, Environment] = {
        # "isaac_sim": IsaacSimEnv,
        "mujoco": MujocoEnv,
        "real_world": NotImplementedError,
    }
    Humans: Dict[str, Human] = {
        "aji5": Aji5,
    }
    Ctx: Dict[str, Context] = {
        "robot_vision_640x480": RobotVision640X480,
        "robot_depth_vision_640x480": RobotDepthVision640X480,
        "robot_joint_mapping": RobotJointMapping,
        "robot_joints": RobotJoints,
        "robot_qpos": RobotQpos,
        "robot_qvel": RobotQvel,
        "robot_imu": RobotImu,
        "robot_contact": RobotContact,
        "robot_efc_force": RobotEfcForce,
        "robot_geom_mapping": RobotGeomMapping,
        "robot_geom_xmat": RobotGeomXmat,
        "robot_geoms": RobotGeoms,
        "robot_body_geoms": RobotBodyGeoms,
        "robot_force": RobotForce,
    }
    Perceptors: Dict[str, Perceptor] = {
        "photoreceptor": Photoreceptor,
        "imu_perceptor": ImuPerceptor,
        "joints_perceptor": JointPerceptor,
        "force_perceptor": ForcePerceptor,
    }
    Cerebrum: Dict[str, torch.nn.Module] = {
        "lstm": LSTM,
        "xlstm": xLSTMCerebrum,
    }
    Amygdala: Dict[str, torch.nn.Module] = {
        "aji5emo": {
            "mcts": MCTS,
            "policy_value_net": PolicyValueNet,
        },
    }
    Constraints: Dict[str, Constraint] = {}
    Instincts: Dict[str, Instinct] = {
        "rooting_reflex": RootingReflex,
        "suck_reflex": SuckReflex,
        "tonic_neck_reflex": TonicNeckReflex,
        "neck_righting_reflex": NeckRightingReflex,
    }
    PlanReceipes: Dict[Tuple[str, str], NeuralPDDLReceipe] = {
        ("yx", "guided_yx"): Grid2DMovementReceipe,
    }
    Planners: Dict[str, Planner] = {
        "downward": DownwardPlanner,
    }

    def __init__(self):
        self.client = httpx.Client()
        self.config_file = os.environ["CONFIG_FILE"]
        self.config = yaml.safe_load(open(self.config_file))

        self.constraints: List[Constraint] = (
            [
                self.Constraints[constraint["name"]](**constraint["configuration"])
                if constraint["configuration"] is not None
                else self.Constraints[constraint["name"]]()
                for constraint in self.config["constraints"]
            ]
            if "constraints" in self.config and self.config["constraints"] is not None
            else []
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env_type = self.config["environment"]["env"]
        human_config = self.config["human"]
        # for human_config in human_configs:
        identifier = human_config["name"]
        self.env: Environment = self.Env[env_type](
            **self.config["environment"]["configuration"]
        )
        context = (
            [
                self.Ctx[context["name"]](
                    identifier=identifier,
                    **context["configuration"],
                )
                if context["configuration"] is not None
                else self.Ctx[context["name"]](
                    identifier=identifier,
                )
                for context in human_config["context"]
            ]
            if "context" in human_config and human_config["context"] is not None
            else []
        )

        perceptors = (
            [
                self.Perceptors[perceptor["name"]](
                    device=self.device, **perceptor["configuration"]
                )
                if perceptor["configuration"] is not None
                else self.Perceptors[perceptor["name"]](device=self.device)
                for perceptor in human_config["perceptors"]
            ]
            if "perceptors" in human_config and human_config["perceptors"] is not None
            else []
        )

        instincts = (
            [
                self.Instincts[instinct["name"]](
                    compute_client=self.client, **instinct["configuration"]
                )
                if instinct["configuration"] is not None
                else self.Instincts[instinct["name"]](
                    compute_client=self.client,
                )
                for instinct in human_config["instincts"]
            ]
            if "instincts" in human_config and human_config["instincts"] is not None
            else []
        )

        instinct_names = [instinct.name for instinct in instincts]

        perception_latent_size = sum(
            [perceptor.latent_size for perceptor in perceptors]
        )
        perception_latent_names = [perceptor.name for perceptor in perceptors]

        if "memory" in human_config and human_config["memory"] is None:
            cerebrum = None
            amygdala = None
            memory = None
        else:
            context_length = human_config["memory"]["context_length"]

            # 大腦
            cerebrum = self.Cerebrum[human_config["memory"]["cerebrum"]](
                modules=human_config["memory"]["modules"],
                context_length=context_length,
                perception_latent_size=perception_latent_size,
                perception_latent_names=perception_latent_names,
                device=self.device,
            )

            # 杏仁核
            amygdala_config = (
                self.Amygdala[human_config["memory"]["amygdala"]]
                if human_config["memory"]["amygdala"] is not None
                else None
            )
            amygdala = None
            if amygdala_config is not None:
                assert (
                    "emotion" in cerebrum.output_modules
                ), "memory must have emotion module"

                policy_value_net = amygdala_config["policy_value_net"](
                    state_dim=len(instincts),
                    instinct_dim=len(instincts),
                    hidden_dim=64,
                )
                amygdala = amygdala_config["mcts"](
                    policy_value_net=policy_value_net,
                    instinct_names=instinct_names,
                    device=self.device,
                )

            memory = Memory(
                id=identifier,
                context_length=context_length,
                cerebrum=cerebrum,
                amygdala=amygdala,
            )

        plan_receipes = {}
        planner_name = None
        planner_config = {}
        planner = None

        if "planning" in human_config and human_config["planning"] is not None:
            if "receipes" in human_config["planning"]:
                for receipe_config in human_config["planning"]["receipes"]:
                    ctx_name = receipe_config["ctx"]
                    guide = receipe_config["guide"]
                    plan_receipes[(ctx_name, guide)] = self.PlanReceipes[
                        (ctx_name, guide)
                    ](**receipe_config["configuration"], agent=identifier)

            planner_name = human_config["planning"]["planner"]["name"]
            planner_config = human_config["planning"]["planner"]["configuration"]

        if planner_name is not None:
            planner = (
                self.Planners[planner_name](
                    plan_receipes=plan_receipes, **planner_config
                )
                if planner_config is not None
                else self.Planners[planner_name](plan_receipes=plan_receipes)
            )

        self.human: Human = self.Humans[identifier](
            perceptors=perceptors,
            instincts=instincts,
            constraints=self.constraints,
            memory=memory,
            context=context,
            planner=planner,
            device=self.device,
        )

        self.human.manifest(self.env)

    def start(self):
        if os.environ["VERBOSE"] == "silent":
            self.env.run_silent()
        else:
            self.env.run()

    def stop(self):
        self.env.deactivate()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="aji5")
    parser.add_argument("-uc", "--unconsciousness", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--silent", action="store_true")

    args = parser.parse_args()
    os.environ["CONFIG_FILE"] = os.path.join(CONFIG_DIR, f"{args.config}.yaml")
    os.environ["DOWNWARD_PATH"] = DOWNWARD_PATH
    os.environ["EXPERIMENT_DIR"] = os.path.join(EXPERIMENT_DIR, args.config)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
    os.environ["UNCONSCIOUS"] = str(args.unconsciousness)
    os.environ["VERBOSE"] = "silent" if args.silent else "verbose"
    os.environ["DEBUG"] = str(args.debug)

    os.makedirs(os.environ["EXPERIMENT_DIR"], exist_ok=True)
    host = Host()
    try:
        host.start()
    except Exception as e:
        traceback.print_exc()
        print(e)
    finally:
        host.stop()
        host.client.close()
        gc.collect()

# flake8: noqa: F403, E741, F405
import gc
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

import torch
import yaml
from fastapi import FastAPI

# from icecream import ic
from prometheus_client import start_http_server

from environment import *
from human.constraints import *
from human.context import *
from human.humans import *
from human.instincts import *
from human.memory.cerebrum import LSTM, xLSTMCerebrum
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
    # Simulators: Dict[str, Simulator] = {
    #     "mujoco": GraceSimulatorMujoco,
    # }
    Env: Dict[str, Environment] = {
        # "isaac_sim": IsaacSimEnv,
        "mujoco": MujocoEnv,
        "real_world": NotImplementedError,
    }
    Humans: Dict[str, Human] = {
        "grace": Grace,
    }
    Ctx: Dict[str, Context] = {
        "robot_vision_640x480": RobotVision640X480,
        "robot_depth_vision_640x480": RobotDepthVision640X480,
        "robot_joints": RobotJoints,
        "robot_imu": RobotImu,
        "robot_contact": RobotContact,
        "robot_efc_force": RobotEfcForce,
        "robot_geom_mapping": RobotGeomMapping,
        "robot_geom_xmat": RobotGeomXmat,
        "robot_geoms": RobotGeoms,
        "robot_ik": RobotIk,
    }
    Perceptors: Dict[str, Perceptor] = {
        "photoreceptor": Photoreceptor,
        "imu_perceptor": ImuPerceptor,
        "joints_perceptor": JointPerceptor,
    }
    Cerebrum: Dict[str, torch.nn.Module] = {
        "lstm": LSTM,
        "xlstm": xLSTMCerebrum,
    }
    Constraints: Dict[str, Constraint] = {}
    Instincts: Dict[str, Instinct] = {
        "rooting_reflex": RootingReflex,
        "suck_reflex": SuckReflex,
        "tonic_neck_reflex": TonicNeckReflex,
    }
    PlanReceipes: Dict[Tuple[str, str], NeuralPDDLReceipe] = {
        ("yx", "guided_yx"): Grid2DMovementReceipe,
    }
    Planners: Dict[str, Planner] = {
        "downward": DownwardPlanner,
    }
    Executors: Dict[str, Executor] = {
        # "ros2": Ros2Executor,
        # "mujoco": MujocoExecutor,
    }
    Ros2Publishers: Dict[str, Any] = {
        # "/joint_command": HumanJointPublisher,
    }
    Ros2Subscribers: Dict[str, Any] = {
        # "/joint_states": HumanJointSubscriber,
        # "/camera/image_raw": HumanVisionSubscriber,
        # "/imu": HumanImuSubscriber,
    }

    def __init__(self):
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
        if env_type == "isaac_sim":
            import rclpy

            rclpy.init(args=None)  # must be called before adding rclpy nodes

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
                self.Instincts[instinct["name"]](**instinct["configuration"])
                if instinct["configuration"] is not None
                else self.Instincts[instinct["name"]]()
                for instinct in human_config["instincts"]
            ]
            if "instincts" in human_config and human_config["instincts"] is not None
            else []
        )

        perception_latent_size = sum(
            [perceptor.latent_size for perceptor in perceptors]
        )
        perception_latent_names = [perceptor.name for perceptor in perceptors]

        if "memory" in human_config and human_config["memory"] is None:
            cerebrum = None
            memory = None
        else:
            context_length = human_config["memory"]["context_length"]
            cerebrum = self.Cerebrum[human_config["memory"]["cerebrum"]](
                modules=human_config["memory"]["modules"],
                context_length=context_length,
                perception_latent_size=perception_latent_size,
                perception_latent_names=perception_latent_names,
                device=self.device,
            )

            memory = Memory(
                id=identifier,
                context_length=context_length,
                cerebrum=cerebrum,
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
        set_host(self)

    def start(self):
        self.env.activate()

    def stop(self):
        self.env.deactivate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    host = Host()
    host.start()
    yield
    host.stop()
    gc.collect()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return str(type(get_host()))


@app.post("/stop")
async def stop():
    get_host().stop()


HOST: Host = None


def set_host(host: Host):
    global HOST
    HOST = host


def get_host() -> Host:
    return HOST


if __name__ == "__main__":
    from argparse import ArgumentParser

    import uvicorn

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="grace")
    parser.add_argument("-uc", "--unconsciousness", action="store_true")

    args = parser.parse_args()
    os.environ["CONFIG_FILE"] = os.path.join(CONFIG_DIR, f"{args.config}.yaml")
    os.environ["DOWNWARD_PATH"] = DOWNWARD_PATH
    os.environ["EXPERIMENT_DIR"] = os.path.join(EXPERIMENT_DIR, args.config)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
    os.environ["UNCONSCIOUSNESS"] = str(args.unconsciousness)

    os.makedirs(os.environ["EXPERIMENT_DIR"], exist_ok=True)

    try:
        # start_http_server(8000)
        uvicorn.run(app, host="0.0.0.0", port=8300)
    except KeyboardInterrupt:
        pass

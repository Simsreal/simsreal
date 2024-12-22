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
from human.memory.cerebrum import LSTM
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
    }
    Perceptors: Dict[str, Perceptor] = {
        "photoreceptor": Photoreceptor,
        "imu_perceptor": ImuPerceptor,
    }
    Constraints: Dict[str, Constraint] = {
        "speed_limit": SpeedLimit,
        "physical_boundary": PhysicalBoundary,
    }
    Instincts: Dict[str, Instinct] = {
        "fear_of_cold": FearOfCold,
    }
    PlanReceipes: Dict[Tuple[str, str], NeuralPDDLReceipe] = {
        ("yx", "guided_yx"): Grid2DMovementReceipe,
    }
    Planners: Dict[str, Planner] = {
        "downward": DownwardPlanner,
    }
    Executors: Dict[str, Executor] = {
        # "ros2": Ros2Executor,
        "mujoco": MujocoExecutor,
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

        # self.humans: List[Human] = []

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
            if "perceptors" in human_config
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
            cerebrum = LSTM(
                modules=human_config["memory"]["modules"],
                hidden_size=human_config["memory"]["hidden_size"],
                num_layers=human_config["memory"]["num_layers"],
                perception_latent_size=perception_latent_size,
                perception_latent_names=perception_latent_names,
                device=self.device,
            )

            memory = Memory(
                id=identifier,
                capacity=human_config["memory"]["memory_capacity"],
                cerebrum=cerebrum,
            )

        plan_receipes = {}
        planner_name = None
        planner_config = {}
        planner = None
        executor = None

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

        if "executor" in human_config and human_config["executor"] is not None:
            executor_name = human_config["executor"]["name"]
            executor_config = {}
            if (
                "configuration" in human_config["executor"]
                and human_config["executor"]["configuration"] is not None
            ):
                executor_config = human_config["executor"]["configuration"]
                publishers = (
                    executor_config.get("publishers", [])
                    if executor_config is not None
                    and "publishers" in executor_config
                    and executor_config["publishers"] is not None
                    else []
                )
                subscribers = (
                    executor_config.get("subscribers", [])
                    if executor_config is not None
                    and "subscribers" in executor_config
                    and executor_config["subscribers"] is not None
                    else []
                )
                publishers_nodes = {}
                subscribers_nodes = {}

                for publisher in publishers:
                    publisher_name = publisher["topic"]
                    publisher_config = publisher["configuration"]
                    publisher_node = (
                        self.Ros2Publishers[publisher_name](
                            **publisher_config,
                            identifier=identifier,
                            publish_data=self.env.publish_data,
                            publish_locks=self.env.publish_locks,
                        )
                        if publisher_config is not None
                        else self.Ros2Publishers[publisher_name](
                            publish_data=self.env.publish_data,
                            publish_locks=self.env.publish_locks,
                            identifier=identifier,
                        )
                    )
                    publishers_nodes[publisher_name] = publisher_node

                executor_config["publishers"] = publishers_nodes

                for subscriber in subscribers:
                    subscriber_name = subscriber["topic"]
                    subscriber_config = subscriber["configuration"]
                    subscriber_node = (
                        self.Ros2Subscribers[subscriber_name](
                            **subscriber_config,
                            identifier=identifier,
                            subscription_data=self.env.subscription_data,
                            subscription_locks=self.env.subscription_locks,
                        )
                        if subscriber_config is not None
                        else self.Ros2Subscribers[subscriber_name](
                            identifier=identifier,
                            subscription_data=self.env.subscription_data,
                            subscription_locks=self.env.subscription_locks,
                        )
                    )
                    subscribers_nodes[subscriber_name] = subscriber_node

                executor_config["subscribers"] = subscribers_nodes

            executor = (
                self.Executors[executor_name](**executor_config)
                if executor_config is not None
                else self.Executors[executor_name]()
            )

        self.human: Human = self.Humans[identifier](
            perceptors=perceptors,
            instincts=instincts,
            constraints=self.constraints,
            memory=memory,
            context=context,
            planner=planner,
            executor=executor,
            device=self.device,
        )

        self.human.manifest(self.env)
        set_host(self)

    def start(self):
        if self.human.executor is not None:
            self.human.executor.start()
        self.env.activate()

    def stop(self):
        if self.human.executor is not None:
            self.human.executor.stop()
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

    args = parser.parse_args()
    os.environ["CONFIG_FILE"] = os.path.join(CONFIG_DIR, f"{args.config}.yaml")
    os.environ["DOWNWARD_PATH"] = DOWNWARD_PATH
    os.environ["EXPERIMENT_DIR"] = os.path.join(EXPERIMENT_DIR, args.config)

    os.makedirs(os.environ["EXPERIMENT_DIR"], exist_ok=True)

    try:
        start_http_server(8000)
        uvicorn.run(app, host="0.0.0.0", port=8300)
    except KeyboardInterrupt:
        pass

# flake8: noqa: F403, E741, F405
import gc
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import rclpy
import torch
import yaml
from fastapi import FastAPI
from icecream import ic
from prometheus_client import start_http_server

from environment import *
from human.constraints import *
from human.context import *
from human.environment.isaac import *
from human.humans import *
from human.instincts import *
from human.neuro_symbol import *
from human.neuro_symbol.downward_planner import DownwardPlanner
from human.neuro_symbol.receipes import *
from human.nn_modules.cerebrum import LSTM
from human.perceptors import *
from intelligence.memory import Memory
from schema.environment import EnvironmentConfig, Landmarks  # for grid2d

CONFIG_DIR = "simulation_config"
DOWNWARD_PATH = "downward/fast-downward.py"


class Host:
    Env: Dict[str, Environment] = {
        "grid2d": Grid2DEnv,
        "isaac_sim": IsaacSimEnv,
        "real_world": NotImplementedError,
    }
    Humans: Dict[str, Human] = {
        "alice": Alice,
        "bob": Bob,
        "charles": Charles,
        "david": David,
        "ella": Ella,
        "felix": Felix,
        "grace": Grace,
    }
    Ctx: Dict[str, Context] = {
        "yx": YX,
    }
    Perceptors: Dict[str, Perceptor] = {
        "grid_vision": GridVision,
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
        "fake": FakeExecutor,
        "ros2": Ros2Executor,
    }
    IsaacPublishers: Dict[str, rclpy.node.Node] = {
        "/joint_command": HumanJointPublisher,
    }
    IsaacSubscribers: Dict[str, rclpy.node.Node] = {
        "/joint_states": HumanJointSubscriber,
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
            rclpy.init(args=None)  # must be called before adding rclpy nodes

        self.env: Environment = self.Env[env_type](
            **self.config["environment"]["configuration"], create=True
        )
        self.humans: List[Human] = []

        human_configs = self.config["humans"]
        for human_config in human_configs:
            perceptors = (
                [
                    self.Perceptors[perceptor["name"]](**perceptor["configuration"])
                    if perceptor["configuration"] is not None
                    else self.Perceptors[perceptor["name"]]()
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

            context = (
                [
                    self.Ctx[context["name"]](**context["configuration"])
                    if context["configuration"] is not None
                    else self.Ctx[context["name"]]()
                    for context in human_config["context"]
                ]
                if "context" in human_config and human_config["context"] is not None
                else []
            )

            ctx_size = sum([ctx.size for ctx in context])
            ctx_names = [ctx.name for ctx in context]

            if "memory" in human_config and human_config["memory"] is None:
                nn_module = None
                memory = None
            else:
                nn_module = LSTM(
                    modules=human_config["memory"]["modules"],
                    hidden_size=human_config["memory"]["hidden_size"],
                    num_layers=human_config["memory"]["num_layers"],
                    ctx_size=ctx_size,
                    ctx_names=ctx_names,
                    device=self.device,
                )

                memory = Memory(
                    id=human_config["name"],
                    capacity=human_config["memory"]["memory_capacity"],
                    nn_module=nn_module,
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
                        ](**receipe_config["configuration"], agent=human_config["name"])

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
                        executor_config["publishers"]
                        if executor_config is not None
                        and "publishers" in executor_config
                        else []
                    )
                    subscribers = (
                        executor_config["subscribers"]
                        if executor_config is not None
                        and "subscribers" in executor_config
                        else []
                    )
                    publishers_nodes = {}
                    subscribers_nodes = {}

                    for publisher in publishers:
                        publisher_name = publisher["topic"]
                        publisher_config = publisher["configuration"]
                        publisher_node = (
                            self.IsaacPublishers[publisher_name](**publisher_config)
                            if publisher_config is not None
                            else self.IsaacPublishers[publisher_name]()
                        )
                        publishers_nodes[publisher_name] = publisher_node

                    executor_config["publishers"] = publishers_nodes

                    for subscriber in subscribers:
                        subscriber_name = subscriber["topic"]
                        subscriber_config = subscriber["configuration"]
                        subscriber_node = (
                            self.IsaacSubscribers[subscriber_name](
                                **subscriber_config,
                                subscription_data=self.env.subscription_data,
                                subscription_locks=self.env.subscription_locks,
                            )
                            if subscriber_config is not None
                            else self.IsaacSubscribers[subscriber_name](
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

            human: Human = self.Humans[human_config["name"]](
                perceptors=perceptors,
                instincts=instincts,
                constraints=self.constraints,
                memory=memory,
                context=context,
                planner=planner,
                executor=executor,
                device=self.device,
            )

            if env_type == "isaac_sim":
                human.manifest(self.env)
            else:
                human_env = self.Env[env_type](
                    **self.config["environment"]["configuration"], create=False
                )
                human.manifest(human_env)

            self.humans.append(human)

        set_host(self)

    def start(self):
        for human in self.humans:
            human.let_be_thread.start()
            human.executor.start()

    def stop(self):
        for human in self.humans:
            human.terminate = True
            human.let_be_thread.join()

            if human.executor is not None:
                human.executor.stop()

        self.env.close()


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


@app.post("/update_env")
async def update_env(env_config: EnvironmentConfig):
    """
    For IsaacSim or real world, environment should not be changed here.
    """
    ic(env_config)
    return "not implemented"


@app.post("/update_landmarks")
async def update_landmarks(landmarks: Landmarks):
    """
    Only used for grid2d
    """
    host = get_host()
    landmarks = landmarks.model_dump()["landmarks"]
    host.env.update_landmarks(landmarks)
    return "ok"


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
    parser.add_argument("--config", type=str, default="grid_2d/felix")
    args = parser.parse_args()
    os.environ["CONFIG_FILE"] = os.path.join(CONFIG_DIR, f"{args.config}.yaml")
    os.environ["DOWNWARD_PATH"] = DOWNWARD_PATH

    try:
        start_http_server(8000)
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        pass

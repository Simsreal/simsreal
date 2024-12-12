# flake8: noqa: F403, E741, F405
import os
from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI
from icecream import ic
from prometheus_client import start_http_server

from environment import *
from human.constraints import *
from human.context import *
from human.humans import *
from human.instincts import *
from human.neuro_symbol.downward_planner import DownwardPlanner
from human.neuro_symbol.fake_executor import FakeExecutor
from human.neuro_symbol.receipes import *
from human.nn_modules.cerebrum import LSTM
from human.perceptors import *
from intelligence.memory import Memory
from schema.environment import EnvironmentConfig, Landmarks

CONFIG_DIR = "simulation_config"
DOWNWARD_PATH = "downward/fast-downward.py"


class Host:
    Env = {"grid2d": Grid2D}
    Humans = {
        "alice": Alice,
        "bob": Bob,
        "charles": Charles,
        "david": David,
        "ella": Ella,
        "felix": Felix,
    }
    Perceptors = {"grid_vision": GridVision}
    Instincts = {"fear_of_cold": FearOfCold}
    Constraints = {"speed_limit": SpeedLimit, "physical_boundary": PhysicalBoundary}
    Ctx = {"yx": YX}
    Planners = {"downward": DownwardPlanner}
    PlanReceipes = {
        ("yx", "guided_yx"): Grid2DMovementReceipe,
    }
    Executors = {"fake": FakeExecutor}

    def __init__(self):
        self.config_file = os.environ["CONFIG_FILE"]
        self.config = yaml.safe_load(open(self.config_file))

        self.constraints = [
            self.Constraints[constraint["name"]](**constraint["configuration"])
            if constraint["configuration"] is not None
            else self.Constraints[constraint["name"]]()
            for constraint in self.config["constraints"]
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = self.Env[self.config["environment"]["env"]](
            **self.config["environment"]["configuration"], create=True
        )
        self.humans = []

        human_configs = self.config["humans"]
        for human_config in human_configs:
            # perceptions
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
                if "instincts" in human_config
                else []
            )

            context = (
                [
                    self.Ctx[context["name"]](**context["configuration"])
                    if context["configuration"] is not None
                    else self.Ctx[context["name"]]()
                    for context in human_config["context"]
                ]
                if "context" in human_config
                else []
            )

            ctx_size = sum([ctx.size for ctx in context])
            ctx_names = [ctx.name for ctx in context]

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

            # planning and executions
            plan_receipes = {}

            if "planning" in human_config and "receipes" in human_config["planning"]:
                for receipe_config in human_config["planning"]["receipes"]:
                    ctx_name = receipe_config["ctx"]
                    guide = receipe_config["guide"]
                    plan_receipes[(ctx_name, guide)] = self.PlanReceipes[
                        (ctx_name, guide)
                    ](**receipe_config["configuration"], agent=human_config["name"])

            if "planning" not in human_config:
                planner_name = "downward"  # default planner
                planner_config = {}
            else:
                planner_name = human_config["planning"]["planner"]["name"]
                planner_config = human_config["planning"]["planner"]["configuration"]

            planner = (
                self.Planners[planner_name](
                    plan_receipes=plan_receipes, **planner_config
                )
                if planner_config is not None
                else self.Planners[planner_name](plan_receipes=plan_receipes)
            )

            executor = self.Executors["fake"]()

            human = self.Humans[human_config["name"]](
                perceptors=perceptors,
                instincts=instincts,
                constraints=self.constraints,
                memory=memory,
                context=context,
                planner=planner,
                executor=executor,
                device=self.device,
            )
            human_env = self.Env[self.config["environment"]["env"]](
                **self.config["environment"]["configuration"], create=False
            )
            human.manifest(human_env)
            self.humans.append(human)

        set_host(self)

    def start(self):
        for human in self.humans:
            human.let_be_thread.start()

    def stop(self):
        for human in self.humans:
            human.terminate = True
            human.let_be_thread.join()

        self.env.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    host = Host()
    host.start()
    yield
    host.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return str(type(get_host()))


@app.post("/update_env")
async def update_env(env_config: EnvironmentConfig):
    ic(env_config)
    return "not implemented"


@app.post("/update_landmarks")
async def update_landmarks(landmarks: Landmarks):
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
    parser.add_argument("--config", type=str, default="poc_grid/felix")
    args = parser.parse_args()
    os.environ["CONFIG_FILE"] = os.path.join(CONFIG_DIR, f"{args.config}.yaml")
    os.environ["DOWNWARD_PATH"] = DOWNWARD_PATH

    try:
        start_http_server(8000)
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        pass

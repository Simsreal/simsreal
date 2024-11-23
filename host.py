# flake8: noqa: F403, E741, F405
import os
from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI
from icecream import ic
from prometheus_client import start_http_server

from human.constraints import *
from human.context import *
from human.humans import *
from human.instincts import *
from human.nn_modules.cerebrum import LSTM
from human.perceptors import *
from human.schema.environment import EnvironmentConfig, Landmarks
from intelligence.environment import Environment
from intelligence.memory import Memory

CONFIG_DIR = "human/simulation_config"


class Host:
    H = {"alice": Alice, "bob": Bob, "charles": Charles, "david": David}
    P = {"grid_vision": GridVision}
    I = {"fear_of_cold": FearOfCold}
    C = {"speed_limit": SpeedLimit, "physical_boundary": PhysicalBoundary}
    Ctx = {"yx": YX}

    def __init__(self):
        self.config_file = os.environ["CONFIG_FILE"]
        self.config = yaml.safe_load(open(self.config_file))

        self.constraints = [
            self.C[constraint["name"]](**constraint["configuration"])
            if constraint["configuration"] is not None
            else self.C[constraint["name"]]()
            for constraint in self.config["constraints"]
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = Environment(self.config["environment"], create=True)
        self.humans = []

        human_configs = self.config["humans"]
        for human_config in human_configs:
            perceptors = (
                [
                    self.P[perceptor["name"]](**perceptor["configuration"])
                    if perceptor["configuration"] is not None
                    else self.P[perceptor["name"]]()
                    for perceptor in human_config["perceptors"]
                ]
                if "perceptors" in human_config
                else []
            )

            instincts = (
                [
                    self.I[instinct["name"]](**instinct["configuration"])
                    if instinct["configuration"] is not None
                    else self.I[instinct["name"]]()
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

            human = self.H[human_config["name"]](
                perceptors=perceptors,
                instincts=instincts,
                constraints=self.constraints,
                memory=memory,
                context=context,
                device=self.device,
            )
            human.manifest(self.env)
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
    host.env.update_landmarks(landmarks.model_dump())
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
    parser.add_argument("--config", type=str, default="fire_in_winter")
    args = parser.parse_args()
    os.environ["CONFIG_FILE"] = os.path.join(CONFIG_DIR, f"{args.config}.yaml")

    try:
        start_http_server(8000)
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        pass

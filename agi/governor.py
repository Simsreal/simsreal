import time
from enum import IntEnum, auto

import torch

from agi.preference.emotions import AlphaSR, PolicyValueNet
from src.utilities.emotion.pad import (
    get_emotion_magnitude,
    get_emotion_reward,
    get_closest_emotion,
)


class HumanStateSymbol(IntEnum):
    FULLNESS = auto()
    AGE = auto()
    EMOTION = auto()


class HumanState(IntEnum):
    FULLNESS = auto()
    AGE = auto()


def fullness_to_symbol(fullness):
    return 1.0


def age_to_symbol(age):
    return 1.0


def governor(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    device = runtime_engine.get_metadata("device")
    intrinsics = cfg["intrinsics"]
    state_dim = 10
    intrinsic_dim = len(intrinsics)

    def augment(state) -> torch.Tensor:
        human_state = runtime_engine.get_shm("human_state")
        emotions = runtime_engine.get_shm("emotions")
        state[HumanStateSymbol.FULLNESS] = fullness_to_symbol(
            human_state[HumanState.FULLNESS]
        )
        state[HumanStateSymbol.AGE] = age_to_symbol(human_state[HumanState.AGE])

        state[HumanStateSymbol.EMOTION] = get_closest_emotion(
            emotions.squeeze(), return_symbol=True
        )
        return state

    policy_value_net = PolicyValueNet(
        state_dim=state_dim,
        policy_dim=intrinsic_dim,
        hidden_dim=64,
    ).to(device)

    alphasr = AlphaSR(policy_value_net, intrinsics, device)
    optimizer = torch.optim.Adam(alphasr.policy_value_net.parameters(), lr=0.001)

    counter = 0
    stream = torch.cuda.Stream(device=device)

    while True:
        with torch.cuda.stream(stream):  # type: ignore
            state = augment(torch.zeros(state_dim, dtype=torch.float32))
            emotions = runtime_engine.get_shm("emotions")

            emotion_strength = get_emotion_magnitude(emotions.squeeze())
            emotion_reward = get_emotion_reward(emotions.squeeze())
            reward = emotion_strength + emotion_reward

            governance = alphasr.forward(
                state,
                reward,
                optimizer,
            )

            with torch.no_grad():
                runtime_engine.update_shm("governance", governance)

            if counter % cfg["mcts"]["decay_period"] == 0:
                alphasr.decay_visits()

            if counter % cfg["mcts"]["prune_period"] == 0:
                alphasr.prune_states()

            if counter > 1e10:
                counter = 0

            counter += 1

        time.sleep(1 / cfg["running_frequency"])

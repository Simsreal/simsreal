# import time
# from enum import IntEnum, auto

import torch

from agi.preference.emotions import AlphaSR, PolicyValueNet
from src.utilities.emotion.pad import (
    get_emotion_magnitude,
    get_emotion_reward,
)
from src.utilities.queues.queue_util import try_get


def governor(runtime_engine):
    def augment(state) -> torch.Tensor:
        return state

    cfg = runtime_engine.get_metadata("config")
    device = runtime_engine.get_metadata("device")
    intrinsics = cfg["intrinsics"]
    state_dim = 10
    intrinsic_dim = len(intrinsics)
    governor_shm = runtime_engine.get_shared_memory("governor")
    motivator_shm = runtime_engine.get_shared_memory("motivator")

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
            emotion = try_get(governor_shm["emotion"], device)
            if emotion is None:
                continue

            emotion_strength = get_emotion_magnitude(emotion.squeeze())
            emotion_reward = get_emotion_reward(emotion.squeeze())
            reward = emotion_strength + emotion_reward

            governance = alphasr.forward(
                state,
                reward,
                optimizer,
            )

            motivator_shm["governance"].put(governance.detach())

            if counter % cfg["mcts"]["decay_period"] == 0:
                alphasr.decay_visits()

            if counter % cfg["mcts"]["prune_period"] == 0:
                alphasr.prune_states()

            if counter > 1e10:
                counter = 0

            counter += 1

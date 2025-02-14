import time
from enum import IntEnum
from enum import auto

import torch

from human.preference.alphaSR import AlphaSR, PolicyValueNet

# from utilities.emotions.pad import get_closest_emotion
from utilities.emotions.pad import get_emotion_magnitude
from utilities.emotions.pad import get_emotion_reward


class GovernorState(IntEnum):
    DUMMY = auto()


def governor_proc(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    device = runtime_engine.get_metadata("device")
    intrinsics = cfg["intrinsics"]
    state_dim = 10
    intrinsic_dim = len(intrinsics)

    def augment(state) -> torch.Tensor:
        # governor_state = runtime_engine.get_shm("governor_state")
        # to implement
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
            governor_state = augment(torch.zeros(state_dim, dtype=torch.float32))
            # print(governor_state)
            emotions = runtime_engine.get_shm("emotions")

            emotion_strength = get_emotion_magnitude(emotions.squeeze())
            emotion_reward = get_emotion_reward(emotions.squeeze())
            reward = emotion_strength + emotion_reward

            governance = alphasr.forward(
                governor_state,
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

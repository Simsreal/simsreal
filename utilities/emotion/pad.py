import numpy as np
import torch

emotion_look_up = {
    "happy": torch.tensor(
        [
            0.80,
            0.60,
            0.50,
        ],
        dtype=torch.float32,
    ),
    "sad": torch.tensor(
        [
            -0.70,
            -0.50,
            -0.40,
        ],
        dtype=torch.float32,
    ),
    "angry": torch.tensor(
        [
            -0.60,
            0.70,
            0.60,
        ],
        dtype=torch.float32,
    ),
    "fearful": torch.tensor(
        [
            -0.75,
            0.70,
            -0.50,
        ],
        dtype=torch.float32,
    ),
    "surprised": torch.tensor(
        [
            0.50,
            0.80,
            0.40,
        ],
        dtype=torch.float32,
    ),
    "disgusted": torch.tensor(
        [
            -0.60,
            0.50,
            0.50,
        ],
        dtype=torch.float32,
    ),
    "neutral": torch.tensor(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    ),
    "tender": torch.tensor(
        [
            0.70,
            0.40,
            0.40,
        ],
        dtype=torch.float32,
    ),
    "excited": torch.tensor(
        [
            0.70,
            0.75,
            0.60,
        ],
        dtype=torch.float32,
    ),
    "confused": torch.tensor(
        [
            -0.20,
            0.30,
            -0.10,
        ],
        dtype=torch.float32,
    ),
    "pleased": torch.tensor(
        [
            0.90,
            0.70,
            0.60,
        ],
        dtype=torch.float32,
    ),
    "satisfied": torch.tensor(
        [
            0.85,
            0.65,
            0.55,
        ],
        dtype=torch.float32,
    ),
    "bored": torch.tensor(
        [
            -0.30,
            -0.20,
            -0.10,
        ],
        dtype=torch.float32,
    ),
}

emotion_id_lookup = {emotion: idx for idx, emotion in enumerate(emotion_look_up.keys())}


def get_closest_emotion(emotion, return_symbol=False):
    closest_emotion = min(
        list(emotion_look_up.keys()),
        key=lambda x: torch.linalg.norm(emotion_look_up[x] - emotion),
    )
    return emotion_id_lookup[closest_emotion] if return_symbol else closest_emotion


def get_emotion_reward(pad_vector):
    happy = emotion_look_up["happy"]
    distance = torch.linalg.norm(pad_vector - happy, ord=2)
    reward = 1.0 / (1.0 + distance.item())
    return reward


def get_emotion_magnitude(pad_vector):
    if isinstance(pad_vector, torch.Tensor):
        magnitude = torch.sqrt(torch.sum(torch.square(pad_vector))).item()
    else:
        magnitude = np.sqrt(np.sum(np.square(pad_vector)))
    return magnitude


if __name__ == "__main__":
    emotion = torch.tensor([[0.99, 0.99, 0.99]])
    print(get_closest_emotion(emotion, return_symbol=False))
    print(get_emotion_reward(emotion))
    print(get_emotion_magnitude(emotion.squeeze().numpy()))

from enum import IntEnum, auto

import torch


class HumanState(IntEnum):
    FULLNESS = auto()
    AGE = auto()


def fullness_to_symbol(fullness):
    # range[0, 1]
    if isinstance(fullness, torch.Tensor):
        fullness = fullness.item()

    assert fullness <= 1 and fullness >= 0, "invalid fullness"

    # if fullness

    return 1.0


def age_to_symbol(age):
    # range[0, ->]
    if isinstance(age, torch.Tensor):
        age = age.item()

    assert age >= 0, "invalid age"

    return 1.0

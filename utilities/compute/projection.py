import numpy as np


def project_to_horizontal(vec: np.ndarray, up_axis="z") -> np.ndarray:
    """
    project the vector on to horizontal plan given by the up axis
    """
    mapping = {"x": 0, "y": 1, "z": 2}
    vector_proj = vec.copy()
    vector_proj[mapping[up_axis]] = 0
    return vector_proj / np.linalg.norm(vector_proj)

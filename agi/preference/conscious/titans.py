import torch
import torch.nn as nn
from titans_pytorch import NeuralMemory

mem = NeuralMemory(
    dim=16,
    chunk_size=64,  # set to smaller chunk size for better perf on smaller sequence lengths (but more memory usage)
).cuda()

for i in range(10):
    seq = torch.randn(2, 1024, 16).cuda()
    retrieved, mem_state = mem(seq)

    print(retrieved[:, -1, :])


class Titans(nn.Module):
    def __init__(self):
        pass

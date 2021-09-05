import numpy as np
from torch.utils.data import DataLoader

from buffers import DynamicReplayBuffer


def test_dynamic_buffer():
    buffer = DynamicReplayBuffer(num_envs=3)
    buffer.allocate(None)
    buffer.push(
        [np.array([3, 2, 1]), np.array([3, 2, 1])],
        [1, 1],
        [1, 1],
        [1, 1]
    )
    return buffer


def test_dataloader_compatibility(buffer):
    data_loader = DataLoader(buffer.data)
    return data_loader

buffer = test_dynamic_buffer()
loader = test_dataloader_compatibility(buffer)

buffer.push(
    [np.array([1, 2, 3]), np.array([1, 2, 3])],
    [2, 2],
    [2, 2],
    [2, 2],
)

buffer.data

for i in loader:
    print(i)

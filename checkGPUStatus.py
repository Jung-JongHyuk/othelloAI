import torch

for i in range(torch.cuda.device_count()):
    print(f"cuda:{i}")
    print(f"allocated: {round(torch.cuda.memory_allocated(i) / 1024 ** 3, 1)}GB")
    print(f"cached: {round(torch.cuda.memory_cached(i) / 1024 ** 3, 1)}GB")
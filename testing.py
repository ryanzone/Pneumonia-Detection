import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)


import time

device = torch.device("cuda")

x = torch.rand(5000, 5000, device=device)
y = torch.rand(5000, 5000, device=device)

start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print("Execution Time:", end - start)
print("Device:", z.device)
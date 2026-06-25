import torch
import time

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

device = torch.device("cuda")

x = torch.randn(8,64,512,512,device=device)
conv = torch.nn.Conv2d(64,64,3,padding=1).to(device)

torch.cuda.synchronize()

start=time.time()

for _ in range(100):
    y=conv(x)

torch.cuda.synchronize()

print("Time:",time.time()-start)
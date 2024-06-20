import math
import time
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig
from dataloader import DataLoaderLite


#---------------------------------------------------------------------------#

if __name__=="__main__":
    # Autodetect CUDA device
    device= "None"
    if torch.cuda.is_available():
        device="cuda"
        torch.cuda.manual_seed(1337)
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device="mps"
    else:
        device="cpu"
        torch.manual_seed(1337)

    print(f"using device {device}")




    train_loader=DataLoaderLite(B=4,T=1024)
    torch.set_float32_matmul_precision('high') # mixed precision training 'high=> TF32'

    #Initialize and load model into the Device
    model = GPT(GPTConfig())
    model.to(device)

    # Optimize
    optimizer= torch.optim.AdamW(model.parameters(), lr=3e-4)

    for i in range(50):
        t0=time.time()
        x,y=train_loader.next_batch()
        x,y= x.to(device), y.to(device)
        optimizer.zero_grad() # Optimizer will accumulate gradients so its necessary to zero out grad for each back prop
        with torch.autocast(device_type=device, dtype=torch.float16)
        logits,loss= model(x,y)
        #import code; code.interact(local=locals())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1=time.time()
        dt=(t1-t0)*1000
        tokens_per_sec= (train_loader.B*train_loader.T)/(t1-t0)
        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")


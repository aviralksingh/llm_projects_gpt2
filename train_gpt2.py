import math
import time
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
from train_config import configure_optimizers


max_lr = 6e-4 # From GPT3 paper
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

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

    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    B = 64 # micro batch size
    T = 1024 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


    train_loader=DataLoaderLite(B=B,T=T)
    torch.set_float32_matmul_precision('high') # mixed precision training 'high=> TF32'

    #Initialize and load model into the Device
    # Increasing the vocab size from 50257 to 50304 to make it power of 2 and ideal for GPU computation.
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model= torch.compile(model)

    # Optimize
    # optimizer= torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8) # From GPT3 paper B: Details of Modelling
    optimizer = configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)



    for step in range(max_steps):
        t0=time.time()
        optimizer.zero_grad() # Optimizer will accumulate gradients so its necessary to zero out grad for each back prop
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


        with torch.autocast(device_type=device, dtype=torch.float16):
            logits,loss= model(x,y)
        #import code; code.interact(local=locals())
        loss.backward()
        norm= torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # From GPT3 paper B: Details of Modelling

        #determine and set the learning rate for this iteration
        lr= get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        torch.cuda.synchronize()
        t1=time.time()
        dt=(t1-t0)*1000
        tokens_per_sec= (train_loader.B*train_loader.T * grad_accum_steps)/(t1-t0)
        print(f"step {i} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


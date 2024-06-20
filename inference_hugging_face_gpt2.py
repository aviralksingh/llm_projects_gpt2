from model import GPT
import torch
from torch.nn import functional as F
import tiktoken

if __name__=="__main__":
   # Autodetect CUDA device
    device= "None"
    if torch.cuda.is_available():
        device="cuda"
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device="mps"
    else:
        device="cpu"
    print(f"using device {device}")

    model= GPT.from_pretrained('gpt2')
    num_of_input_sequence=5
    max_length=30

    model.eval()
    model.to(device)
    print("Loaded the Model")

    enc=tiktoken.get_encoding('gpt2')
    tokens=enc.encode("Hello, I'm a language model,") # Results in 8 tokens check https://tiktokenizer.vercel.app/?model=gpt2
    tokens=torch.tensor(tokens, dtype=torch.long) #(8,)
    tokens=tokens.unsqueeze(0).repeat(num_of_input_sequence , 1) # (5,8)
    x= tokens.to(device)


    # generate! right now x is (B,T) where B=5, T=8
    # set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():  # This code just tells pytorch to not store/cache tensors and gradients
            logits,_= model(x) # (B, T, vocab_size)
            # take the logits only at the last position
            logits= logits[:, -1, :] # Time dimension -1 (last location)
            # get the probabilities
            probs= F.softmax (logits, dim=-1)
            # do top-k sampling of 50 hugging face pipeline default
            # topk_probs here becomes (5,50), topk_indices is (5,50)
            topk_probs, topk_indices= torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs,1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x= torch.cat((x,xcol), dim=1)

# print the generated text
for i in range(num_of_input_sequence):
    tokens= x[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">", decoded)


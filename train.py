from dataclasses import dataclass

import torch
import tqdm
import matplotlib.pyplot as plt

from transformer import Transformer

@dataclass
class Config:
    # Training config
    batch_size: int = 64
    num_iterations: int = 5000
    lr: float = 3e-4
    eval_interval: int = 100

    # Model config
    block_size: int = 12
    vocab_size: int = 65 # Shakespeare dataset
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
    dropout: int = 0.1

def tokenize(text: list):
    
    chars = set(text)
    vocab_size = len(chars)
    # print("Vocab Size: ", vocab_size)

    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}

    encoder = lambda s: [stoi[c] for c in s]
    decoder = lambda l: ''.join([itos[i] for i in l])

    return vocab_size, encoder, decoder

def create_dataset(text: list, encoder, split: float= 0.9):
    dataset = torch.tensor(encoder(text), dtype=torch.long)

    train_size = int(split * len(dataset))
    return dataset[:train_size], dataset[train_size:] # Train/Val split TODO: make this train/val/split

def get_batch(dataset, config):
    ix = torch.randint(len(dataset) - config.block_size, (config.batch_size,))

    x = torch.stack([dataset[i:i+config.block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+config.block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, train, val, config, eval_iters=50):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(train, config) if split == "train" else get_batch(val, config)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, train, val, config, model_loc="./model/transformer.pt"):
    results = {
        "train": [],
        "val": []
    }
    
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    pbar = tqdm.trange(config.num_iterations)
    for step in pbar:
        xb, yb = get_batch(train, config)
        logits, loss = model(xb, yb)

        if step % config.eval_interval == 0:
            l = estimate_loss(model, train, val, config)
            t_train, t_val = l["train"], l["val"]
            results["train"].append(t_train)
            results["val"].append(t_val)
            pbar.set_description(f"Estimated losses: {t_train:0.5f} train, {t_val:0.5f} val")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    
    torch.save(model, model_loc)
    return model, results

def sample_from_model(model, length, decoder):
    context = torch.zeros((1,1), dtype=torch.long)
    return decoder(model.generate(context, max_new_tokens=length)[0].tolist())

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loc', type=str, help="Location of text data, .txt format.")

    args = parser.parse_args()
    with open(args.data_loc, 'r', encoding="utf-8") as f:
        text = f.read()

    
    config = Config()
    vocab_size, encoder, decoder = tokenize(text)
    config.vocab_size = vocab_size

    print("Training config: ", config)
    print("Dataset location: ", args.data_loc)

    train_d, val_d = create_dataset(text, encoder)
    print("Training set size: ", len(train_d))
    print("Validation set size: ", len(val_d))
    
    model = Transformer(config)

    generated = sample_from_model(model, 50, decoder)
    print("\nFresh model sample:\n", generated)

    print("\nTraining model...")
    model, results = train(model, train_d, val_d, config)

    print("\nTrained sample:\n ", sample_from_model(model, 100, decoder))

    plt.plot(results["train"])
    plt.plot(results["val"])
    plt.legend(["Train", "Validation"])
    plt.savefig("./model/results.png")

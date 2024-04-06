import torch
import torch.nn.functional as F
import random

words = open('names.txt', 'r').read().splitlines()

# build maps to and from int
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

# create datasets
block_size = 3
def build_dataset(words):
    X=[]
    Y=[]
    for w in words:
        context = [0] * block_size
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:]+[ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])


# init params
def init_params():
    global C,W1,b1,W2,b2,parameters
    n_embed = 10
    n_hidden = 200

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, n_embed), generator=g)
    W1 = torch.randn((n_embed*block_size, n_hidden), generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn((n_hidden, vocab_size), generator=g)
    b2= torch.randn(vocab_size, generator=g)

    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

init_params()

# train
def mlp():
    max_steps = 600000
    batch_size = 32

    for i in range(max_steps):
        # forward
        ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        preact = embcat @ W1 + b1
        h = torch.tanh(preact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None

        loss.backward()

        lr = 0.1 if i < 100000 else 0.1
        for p in parameters:
            p.data += -lr * p.grad

mlp()

def split_loss(split):
    Xsplit, Ysplit = {
        'train': (Xtr, Ytr),
        'dev': (Xdev, Ydev),
        'test': (Xtest, Ytest),
    }[split]
    emb = C[Xsplit]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Ysplit)

print(split_loss('train'))

def sample():
    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context  = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))

sample()
# sample

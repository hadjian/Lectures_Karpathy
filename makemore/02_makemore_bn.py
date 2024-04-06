import torch
import numpy as np
import torch.nn.functional as F
import plotille as plt
import random


class Linear:    
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out) / fan_in**0.5
        self.bias = torch.randn(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.001):
        self.eps = eps
        self.m = momentum
        self.training = True
        self.gamma = torch.ones((1, dim))
        self.beta = torch.zeros((1, dim))
        self.running_mean = torch.zeros((1, dim))
        self.running_var = torch.ones((1, dim))


    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.std(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1.0-self.m) * self.running_mean + self.m * xmean
                self.running_var = (1.0-self.m) * self.running_var + self.m * xvar            
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        self.out
        return self.out

    def parameters(self):
        return []

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

n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), Tanh(),
    Linear(           n_hidden, n_hidden, bias=False), Tanh(),
    Linear(           n_hidden, vocab_size, bias=False),
]

with torch.no_grad():
    layers[-1].weight *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# train
max_steps = 10000
def mlp():
    batch_size = 32
    global lossi, ud
    lossi = []
    ud = []

    for i in range(max_steps):
        # forward pass
        ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb = C[Xb]
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb)

        # backward pass
        for layer in layers:
            layer.out.retain_grad()

        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.1
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        if i%1000 == 0:
            print(f"loss at iteration {i}: {loss}")
        lossi.append(loss.log10().item())
        with torch.no_grad():
            ud.append([(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters])
        
        if i > 1000:
            break

mlp()

# visualize activation histograms
fig = plt.Figure()
fig.width = 100
fig.height = 20
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        print('layer %d  (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        hy, hx = torch.histogram(t, density=True)
        fig.plot(hx[:-1].detach().tolist(), hy.detach().tolist(), label=f'layer {i} ({layer.__class__.__name__}')

print(fig.show(legend=True))

# visualize gradient histograms
fig = plt.Figure()
fig.width = 100
fig.height = 20
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        print('layer %d  (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
        hy, hx = torch.histogram(t, density=True)
        fig.plot(hx[:-1].detach().tolist(), hy.detach().tolist(), label=f'layer {i} ({layer.__class__.__name__}')

print(fig.show(legend=True))


# visualize parameter histograms
fig = plt.Figure()
fig.width = 100
fig.height = 20
for i, p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std()/p.std()))
        hy, hx = torch.histogram(t, density=True)
        fig.plot(hx[:-1].detach().tolist(), hy.detach().tolist(), label=f'layer {i} ({layer.__class__.__name__}')

print(fig.show(legend=True))

# visualize data to gradient ratio
fig = plt.Figure()
fig.width = 100
fig.height = 20
for i,p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:
        fig.plot(np.arange(0, len(ud), 1.0), [ud[j][i] for j in range(len(ud))])
fig.plot(np.arange(0, len(ud)), [-3 for i in range(len(ud))], label='k', lc='blue')

print(fig.show(legend=True))

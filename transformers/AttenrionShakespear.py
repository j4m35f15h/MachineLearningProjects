# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:20:46 2024

Decoder-only text generator trained on Shakespear. Uses character level tokens.
After training, sampling the model generates garbelled characters in the format
of a script. Hardware limitations (CPU only) prevented further refinement. To
improve the text generation, the number of layers needs to be increased. This
allows the model to interpret richer levels of information. Additionally, a
tokeniser should be used, though this increaases the computations required.
A character pair encoding may provide superior word generation.

@author: James
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


#Global Params--------------------------------------------------
blockSize = 256
torch.manual_seed(1337)
batchSize = 32
max_iters = 5000
eval_interval = 500
learning_rate =  3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192
n_head = 6
n_layer = 2
dropout = 0.2

#Data input and encode table creation----------------------------
file = open("input.txt")
text = file.read()
chars = sorted(set(text))
vocabSize = len(chars)

stoi,itos = dict(),dict()
for i,ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#Batch Creation function and example ---------------------------    

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = trainData if split == 'train' else testData
    ix = torch.randint(len(data) - blockSize,(batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i+1:i+blockSize+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    
    model.eval()
    
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    
    return out

#Codebook vs sequence length tradeoff: spends more time in the decoding stage for dev with more complex tokenisers


#testTrain Split ----------------------------------------
data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9*len(data))
trainData = data[:n]
testData  = data[n:]



#Defining and creating the model-----------------------
class Head(nn.Module):
    
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias = False)
        self.query  = nn.Linear(n_embd,head_size,bias = False)
        self.value  = nn.Linear(n_embd,head_size,bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(blockSize,blockSize)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)#(B,T,C)
        q = self.query(x) #(B,T,C)
        v = self.value(x)#(B,T,C)
        
        wei = q @ k.transpose(-2,-1) * C**-0.5   #(B,T,C) * (B,C,T) = (B,T,T)
        tril = torch.tril(torch.ones(T,T))
        wei = wei.masked_fill(tril==0,float('-inf'))
        wei = F.softmax(wei,dim = -1)
        wei = self.dropout(wei)
        
        out = wei @ v #(T,T) * (B,T,C) = (B,T,C)
        
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)
        out = self.dropout(self.proj(out))
        return out
        
class FeedForward(nn.Module)  :
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)
 
class Block(nn.Module):
    #The attention gathers interesting information,
    #The FF can make more complex decisions on what to do with the information
    #Combined, they perfrom communication followed by computation
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
        
       
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(vocabSize,n_embd)
        self.position_embedding_table = nn.Embedding(blockSize,n_embd)
        
        self.blocks = nn.Sequential(
            *[Block(n_embd,n_head = n_head) for _ in range(n_layer)]
                                      )
        self.ln_b = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabSize)
        
    def forward(self, idx,targets = None):
        B,T = idx.shape
        
        tok_emb = self.tokenEmbeddingTable(idx) #B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T,device = device))#T,C
        x = tok_emb + pos_emb # BTC
        x = self.blocks(x)
        x = self.ln_b(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits, loss
    
    def generate(self,idx,maxNewTokens):
        for _ in range(maxNewTokens):
            
            idx_cond = idx[:,-blockSize:]
            
            logits,loss = self(idx_cond)
            
            logits = logits[:,-1,:]
            
            probs = F.softmax(logits,dim = -1)
            
            idxNext = torch.multinomial(probs,num_samples = 1)
            
            idx = torch.cat((idx,idxNext),dim = 1)
        return idx




#Model opbject and Parameter optimiser used-----------------------------
model = BigramLanguageModel()
m = model.to(device)
optimiser = torch.optim.AdamW(m.parameters(),lr = learning_rate) #3e-4 used for larger models popularly


#Train Loop-----------------------------------------------------------
for steps in range(max_iters):
    
    xb,yb = get_batch('train')
    
    logits, loss = m(xb,yb)
    
    optimiser.zero_grad(set_to_none = True)
    loss.backward()
    optimiser.step()
    if steps % 100 == 0:
        print("Step {}".format(steps))
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print("Step {}: train Loss = {:.4f}; val loss = {:.4f}".format(steps,losses['train'],losses['val']))


#Sampleing the network------------------------------
idx = torch.zeros((1,1),dtype=torch.long,device = device)
print(decode(m.generate(idx,500)[0].tolist()))


        
        














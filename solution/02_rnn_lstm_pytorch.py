import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from config import DATA_PATH, SAMPLE_SIZE, SEED, MAX_LEN, VOCAB_SIZE, EMBED_DIM, BATCH_SIZE, EPOCHS_RNN, DEVICE
from preprocess import load_and_split

os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)

class Vocab:
    def __init__(self, texts, max_vocab=10000):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        self.itos = ['<pad>', '<unk>'] + [w for w,c in counter.most_common(max_vocab-2)]
        self.stoi = {w:i for i,w in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
    def encode(self, text, max_len=40):
        tokens = text.split()
        ids = [self.stoi.get(tok, self.stoi['<unk>']) for tok in tokens][:max_len]
        if len(ids) < max_len:
            ids += [self.stoi['<pad>']] * (max_len - len(ids))
        return ids

class TextDataset(Dataset):
    def __init__(self, X, y, vocab, max_len):
        self.X = X; self.y = y; self.vocab = vocab; self.max_len = max_len
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        ids = self.vocab.encode(self.X[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        emb = self.embedding(x)
        out, h_n = self.rnn(emb)
        h = h_n.squeeze(0)
        h = self.dropout(h)
        return torch.sigmoid(self.fc(h)).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, 1)
    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, c_n) = self.lstm(emb)
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        h = self.dropout(h)
        return torch.sigmoid(self.fc(h)).squeeze(1)

def train_model(model, train_loader, val_loader, epochs=8, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    for ep in range(epochs):
        model.train()
        train_loss=0; correct=0; total=0
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out,yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*xb.size(0)
            pred = (out>=0.5).float()
            correct += (pred==yb).sum().item()
            total += xb.size(0)
        train_loss/=total; train_acc=correct/total
        model.eval()
        val_loss=0; correct=0; total=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out,yb)
                val_loss += loss.item()*xb.size(0)
                pred = (out>=0.5).float()
                correct += (pred==yb).sum().item()
                total += xb.size(0)
        val_loss/=total; val_acc=correct/total
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        print(f'Epoch {ep+1}/{epochs} train_loss={train_loss:.4f} acc={train_acc:.4f} val_loss={val_loss:.4f} acc={val_acc:.4f}')
    return model, history

def eval_model(model, test_loader):
    model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            pred = (out>=0.5).float().cpu().numpy()
            preds.extend(pred); trues.extend(yb.numpy())
    preds=np.array(preds); trues=np.array(trues)
    acc = accuracy_score(trues,preds)
    prec,rec,f1,_ = precision_recall_fscore_support(trues,preds,average='binary')
    cm = confusion_matrix(trues,preds)
    return acc,prec,rec,f1,cm

def main():
    X_train,X_val,X_test,y_train,y_val,y_test = load_and_split(DATA_PATH, SAMPLE_SIZE, seed=SEED)
    vocab = Vocab(X_train, max_vocab=VOCAB_SIZE)
    print('Vocab size:', vocab.vocab_size)
    train_ds = TextDataset(X_train,y_train,vocab,MAX_LEN)
    val_ds = TextDataset(X_val,y_val,vocab,MAX_LEN)
    test_ds = TextDataset(X_test,y_test,vocab,MAX_LEN)
    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE)
    
    results=[]
    # SimpleRNN
    print('\n=== SimpleRNN ===')
    rnn = SimpleRNN(vocab.vocab_size, EMBED_DIM, 64)
    t0=time.time()
    rnn,hist_rnn = train_model(rnn,train_loader,val_loader,epochs=EPOCHS_RNN)
    t1=time.time()
    acc,p,r,f1,cm = eval_model(rnn,test_loader)
    print(acc,p,r,f1); print(cm)
    results.append({'model':'RNN_Simple','representation':'Sequence+Embedding','accuracy':acc,'precision':p,'recall':r,'f1':f1,'train_time':t1-t0})
    # plot
    plt.figure()
    plt.plot(hist_rnn['train_loss'],label='train'); plt.plot(hist_rnn['val_loss'],label='val'); plt.legend(); plt.title('SimpleRNN Loss'); plt.savefig('../figures/rnn_loss.png'); plt.close()
    plt.figure()
    plt.plot(hist_rnn['train_acc'],label='train'); plt.plot(hist_rnn['val_acc'],label='val'); plt.legend(); plt.title('SimpleRNN Acc'); plt.savefig('../figures/rnn_acc.png'); plt.close()
    
    # LSTM
    print('\n=== LSTM ===')
    lstm = LSTMModel(vocab.vocab_size, EMBED_DIM, 64, bidirectional=False)
    t0=time.time()
    lstm,hist_lstm = train_model(lstm,train_loader,val_loader,epochs=EPOCHS_RNN)
    t1=time.time()
    acc,p,r,f1,cm = eval_model(lstm,test_loader)
    print(acc,p,r,f1); print(cm)
    results.append({'model':'LSTM','representation':'Sequence+Embedding','accuracy':acc,'precision':p,'recall':r,'f1':f1,'train_time':t1-t0})
    plt.figure()
    plt.plot(hist_lstm['train_loss'],label='train'); plt.plot(hist_lstm['val_loss'],label='val'); plt.legend(); plt.title('LSTM Loss'); plt.savefig('../figures/lstm_loss.png'); plt.close()
    plt.figure()
    plt.plot(hist_lstm['train_acc'],label='train'); plt.plot(hist_lstm['val_acc'],label='val'); plt.legend(); plt.title('LSTM Acc'); plt.savefig('../figures/lstm_acc.png'); plt.close()
    
    # BiLSTM
    print('\n=== BiLSTM ===')
    bilstm = LSTMModel(vocab.vocab_size, EMBED_DIM, 64, bidirectional=True)
    t0=time.time()
    bilstm,hist_bilstm = train_model(bilstm,train_loader,val_loader,epochs=EPOCHS_RNN)
    t1=time.time()
    acc,p,r,f1,cm = eval_model(bilstm,test_loader)
    print(acc,p,r,f1); print(cm)
    results.append({'model':'BiLSTM','representation':'Sequence+Embedding','accuracy':acc,'precision':p,'recall':r,'f1':f1,'train_time':t1-t0})
    plt.figure()
    plt.plot(hist_bilstm['train_loss'],label='train'); plt.plot(hist_bilstm['val_loss'],label='val'); plt.legend(); plt.title('BiLSTM Loss'); plt.savefig('../figures/bilstm_loss.png'); plt.close()
    plt.figure()
    plt.plot(hist_bilstm['train_acc'],label='train'); plt.plot(hist_bilstm['val_acc'],label='val'); plt.legend(); plt.title('BiLSTM Acc'); plt.savefig('../figures/bilstm_acc.png'); plt.close()
    
    pd.DataFrame(results).to_csv('../results/rnn_lstm_results.csv', index=False)
    print('Saved results')

if __name__=='__main__':
    main()

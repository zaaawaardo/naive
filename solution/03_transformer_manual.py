import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from config import DATA_PATH, SAMPLE_SIZE, SEED, MAX_LEN, EPOCHS_TRANSFORMER, DEVICE
from preprocess import load_and_split

os.makedirs('../results', exist_ok=True)

MODEL_NAME = 'distilbert-base-uncased'

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def main():
    X_train,X_val,X_test,y_train,y_val,y_test = load_and_split(DATA_PATH, SAMPLE_SIZE, seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    
    train_ds = SentimentDataset(X_train,y_train,tokenizer,MAX_LEN)
    val_ds = SentimentDataset(X_val,y_val,tokenizer,MAX_LEN)
    test_ds = SentimentDataset(X_test,y_test,tokenizer,MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    t0 = time.time()
    for ep in range(EPOCHS_TRANSFORMER):
        model.train()
        total_loss=0; total=0
        for i,batch in enumerate(train_loader):
            if i>200:  # limit for speed on CPU
                break
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*input_ids.size(0)
            total += input_ids.size(0)
        print(f'Epoch {ep+1} loss {total_loss/total:.4f}')
    t1 = time.time()
    print('Train time:', t1-t0)
    
    model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(out.logits, dim=1).cpu().numpy()
            preds.extend(pred); trues.extend(labels.cpu().numpy())
    preds=np.array(preds); trues=np.array(trues)
    acc = accuracy_score(trues,preds)
    prec,rec,f1,_ = precision_recall_fscore_support(trues,preds,average='binary')
    cm = confusion_matrix(trues,preds)
    print(acc,prec,rec,f1)
    print(cm)
    print(classification_report(trues,preds))
    pd.DataFrame([{'model':'Transformer','representation':'Subword+Attention','accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'train_time':t1-t0}]).to_csv('../results/transformer_results.csv', index=False)
    
    test_sents = [
        "I don't think this movie is not bad at all",
        "It's not great, but I don't hate it",
        "Yeah right, this is AMAZING... not",
        "I really wanted to like it, but it was just painfully boring",
        "Well, that was just perfect... said no one ever"
    ]
    for s in test_sents:
        enc = tokenizer(s, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            pred = torch.argmax(out.logits, dim=1).item()
        print('SENT:', s, '->', 'POS' if pred==1 else 'NEG')

if __name__=='__main__':
    main()

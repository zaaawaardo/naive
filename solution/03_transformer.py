import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from config import DATA_PATH, SAMPLE_SIZE, SEED, MAX_LEN, EPOCHS_TRANSFORMER
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

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    prec,rec,f1,_ = precision_recall_fscore_support(labels,preds,average='binary')
    return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}

def main():
    X_train,X_val,X_test,y_train,y_val,y_test = load_and_split(DATA_PATH, SAMPLE_SIZE, seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    train_ds = SentimentDataset(X_train,y_train,tokenizer,MAX_LEN)
    val_ds = SentimentDataset(X_val,y_val,tokenizer,MAX_LEN)
    test_ds = SentimentDataset(X_test,y_test,tokenizer,MAX_LEN)
    
    args = TrainingArguments(
        output_dir='../results/transformer_ckpt',
        num_train_epochs=EPOCHS_TRANSFORMER,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy='no',
        seed=SEED,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    
    t0=time.time()
    trainer.train()
    t1=time.time()
    print('Train time:', t1-t0)
    
    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=1)
    acc = accuracy_score(y_test,y_pred)
    prec,rec,f1,_ = precision_recall_fscore_support(y_test,y_pred,average='binary')
    cm = confusion_matrix(y_test,y_pred)
    print(acc,prec,rec,f1)
    print(cm)
    print(classification_report(y_test,y_pred))
    pd.DataFrame([{'model':'Transformer','representation':'Subword+Attention','accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'train_time':t1-t0}]).to_csv('../results/transformer_results.csv', index=False)
    
    # Qualitative
    test_sents = [
        "I don't think this movie is not bad at all",
        "It's not great, but I don't hate it",
        "Yeah right, this is AMAZING... not",
        "I really wanted to like it, but it was just painfully boring",
        "Well, that was just perfect... said no one ever"
    ]
    for s in test_sents:
        enc = tokenizer(s, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        enc = {k:v.to(model.device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            pred = torch.argmax(out.logits, dim=1).item()
        print('SENT:', s, '->', 'POS' if pred==1 else 'NEG')

if __name__=='__main__':
    main()

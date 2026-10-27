import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

NEGATION_WORDS = {'not', "n't", 'no', 'never', 'none', 'nothing', 'nowhere', 'hardly', 'scarcely', "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't", "can't", "couldn't", "shouldn't", "mightn't", "mustn't"}
STOPWORDS = set(stopwords.words('english')) - NEGATION_WORDS

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
RT_RE = re.compile(r'^\s*RT\s+')
NON_ALPHANUM = re.compile(r'[^a-zA-Z\s]')
MULTISPACE = re.compile(r'\s+')

def clean_tweet(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = RT_RE.sub('', text)
    # Keep hashtag text
    text = HASHTAG_RE.sub(r'\1', text)
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = NON_ALPHANUM.sub(' ', text)
    text = MULTISPACE.sub(' ', text).strip()
    return text

def load_and_split(data_path, sample_size=80000, test_size=0.15, val_size=0.15, seed=1234):
    # Columns: polarity, id, date, query, user, text
    df = pd.read_csv(data_path, encoding='latin-1', header=None,
                     names=['polarity', 'id', 'date', 'query', 'user', 'text'])
    df['label'] = df['polarity'].apply(lambda x: 0 if x == 0 else 1)
    # Stratified sample for balance/speed
    df_pos = df[df['label']==1]
    df_neg = df[df['label']==0]
    n_per_class = sample_size // 2
    df_pos_s = df_pos.sample(n=min(n_per_class, len(df_pos)), random_state=seed)
    df_neg_s = df_neg.sample(n=min(n_per_class, len(df_neg)), random_state=seed)
    df_s = pd.concat([df_pos_s, df_neg_s]).sample(frac=1, random_state=seed).reset_index(drop=True)
    df_s['clean_text'] = df_s['text'].apply(clean_tweet)
    # Remove empty
    df_s = df_s[df_s['clean_text'].str.len() > 0].reset_index(drop=True)
    X = df_s['clean_text'].tolist()
    y = df_s['label'].values.astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, stratify=y, random_state=seed)
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_ratio, stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

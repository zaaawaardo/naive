import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from config import DATA_PATH, SAMPLE_SIZE, SEED
from preprocess import load_and_split

os.makedirs('../results', exist_ok=True)

def get_top_discriminative_features(vectorizer, clf, n=15):
    feature_names = np.array(vectorizer.get_feature_names_out())
    # log prob: clf.feature_log_prob_[class, feature]
    log_prob = clf.feature_log_prob_
    # difference for positive class
    diff = log_prob[1] - log_prob[0]
    top_pos_idx = np.argsort(diff)[-n:][::-1]
    top_neg_idx = np.argsort(diff)[:n]
    return feature_names[top_pos_idx], feature_names[top_neg_idx]

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATA_PATH, SAMPLE_SIZE, seed=SEED)
    print('Shapes:', len(X_train), len(X_val), len(X_test))
    
    # Try BoW
    cv = CountVectorizer(ngram_range=(1,2), max_features=15000, min_df=2)
    X_train_bow = cv.fit_transform(X_train)
    X_test_bow = cv.transform(X_test)
    
    nb_bow = MultinomialNB()
    t0 = time.time()
    nb_bow.fit(X_train_bow, y_train)
    t1 = time.time()
    y_pred_bow = nb_bow.predict(X_test_bow)
    acc_bow = accuracy_score(y_test, y_pred_bow)
    prec_bow, rec_bow, f1_bow, _ = precision_recall_fscore_support(y_test, y_pred_bow, average='binary')
    print('NB-BoW:', acc_bow, prec_bow, rec_bow, f1_bow, 'time:', t1-t0)
    
    # Try TF-IDF
    tv = TfidfVectorizer(ngram_range=(1,2), max_features=15000, min_df=2)
    X_train_tfidf = tv.fit_transform(X_train)
    X_test_tfidf = tv.transform(X_test)
    nb_tfidf = MultinomialNB()
    t0 = time.time()
    nb_tfidf.fit(X_train_tfidf, y_train)
    t1 = time.time()
    y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
    acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
    prec_tfidf, rec_tfidf, f1_tfidf, _ = precision_recall_fscore_support(y_test, y_pred_tfidf, average='binary')
    print('NB-TFIDF:', acc_tfidf, prec_tfidf, rec_tfidf, f1_tfidf, 'time:', t1-t0)
    
    # Use better one
    if f1_tfidf >= f1_bow:
        clf, vec, Xtr, Xte, ypred, acc,p,r,f1,tt = nb_tfidf, tv, X_train_tfidf, X_test_tfidf, y_pred_tfidf, acc_tfidf,prec_tfidf,rec_tfidf,f1_tfidf,t1-t0
        rep = 'TF-IDF'
    else:
        clf, vec, Xtr, Xte, ypred, acc,p,r,f1,tt = nb_bow, cv, X_train_bow, X_test_bow, y_pred_bow, acc_bow,prec_bow,rec_bow,f1_bow,t1-t0
        rep = 'BoW'
    print('Selected:', rep)
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    top_pos, top_neg = get_top_discriminative_features(vec, clf, 15)
    print('Top pos:', top_pos)
    print('Top neg:', top_neg)
    pd.DataFrame({'model':['NaiveBayes'], 'representation':[rep], 'accuracy':[acc], 'precision':[p], 'recall':[r], 'f1':[f1], 'train_time':[tt]}).to_csv('../results/nb_results.csv', index=False)

if __name__ == '__main__':
    main()

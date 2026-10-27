# TP4: Sentiment140 Classification (Naive Bayes, RNN, LSTM, Transformers)

## Setup
```bash
cd tp4
source env/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn nltk tqdm wordcloud torch transformers datasets accelerate
```

## Data
Place Sentiment140 CSV at `data/training.1600000.processed.noemoticon.csv` (already present).

## Run (GPU/CPU)
```bash
cd solution
python 01_naive_bayes.py
python 02_rnn_lstm_pytorch.py
python 03_transformer.py
python 04_evaluate_and_report.py
```

Results saved to `results/`, figures to `figures/`.

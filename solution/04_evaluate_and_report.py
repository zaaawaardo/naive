import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)

def main():
    dfs = []
    for fn in ['nb_results.csv','rnn_lstm_results.csv','transformer_results.csv']:
        path = os.path.join('../results', fn)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if dfs:
        res = pd.concat(dfs, ignore_index=True)
        res.to_csv('../results/all_results.csv', index=False)
        print(res)
        # plot
        plt.figure(figsize=(10,6))
        sns.barplot(data=res, x='model', y='f1')
        plt.title('F1 Score Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../figures/f1_comparison.png')
        plt.close()
        plt.figure(figsize=(10,6))
        sns.barplot(data=res, x='model', y='train_time')
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../figures/time_comparison.png')
        plt.close()
    else:
        print('No results found')

if __name__=='__main__':
    main()

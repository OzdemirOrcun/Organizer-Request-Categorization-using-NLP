import seaborn as sns
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_palette("pastel")


score_files = os.listdir('./scores/ML')
ml_scores_dfs = []
dl_scores_dfs = []
type = ['word','char','char_wb']
#for file in score_files:

for t in range(0,3):
    for n in range(1, 7):
        gatherer = pd.read_excel('./scores/ML/scores_ML_{type}_{n}_gram.xlsx'.format(type = type[t], n = n,),skiprows=0,skip_blank_lines=True)
        ml_scores_dfs.append(gatherer)



ml_scores = pd.concat(ml_scores_dfs,axis = 0,sort=False,join='outer')
ml_scores=ml_scores.loc[:, (ml_scores != 0).any(axis=0)]



word_scores = ml_scores.iloc[:6:]
char_scores = ml_scores.iloc[6:12:]
char_wb_scores = ml_scores.iloc[12:18:]


dl_algos = ['ANN','CNN','RNN+CNN','RNN','OrcunW2VRNN','PreTrainedRNN']

for a in range(0,6):
    lst = pd.read_excel('./scores/DL/scores_DL_{type}.xlsx'.format(type = dl_algos[a]),skiprows=0,skip_blank_lines = True)
    dl_scores_dfs.append(lst)

dl_scores = pd.concat(dl_scores_dfs,axis = 1,sort=False)
dl_scores=dl_scores.loc[:, (dl_scores != 0).any(axis=0)]
dl_scores = dl_scores.dropna(axis=0,how="all")


def plotBars(df,type,algorithm):

    if(algorithm == 'ML'):
        ax = df.plot(kind='bar',
                    figsize=(16, 5),
                    color=['red','blue','green','yellow','purple','orange','gray'],
                    alpha=0.75,
                    rot=45,
                    fontsize=13)

        ax.set_title("F1 Acc. Scores ML İçin {type} N-Gram (1'den 6'ya)".format(type = type))

        for i in ax.patches:
            ax.annotate(str(round(i.get_height(), 3)),
                        (i.get_x() + 0.1, i.get_height() * 1.002), color='black', fontsize=8)

        plt.legend(loc='lower right')

    elif(algorithm == 'DL' and (type == 'None' or type == 'none')):

        mapping_name = dict(zip(list(df.columns),dl_algos))
        df = df.rename(columns = mapping_name)
        df = df[dl_algos]
        df = df.T

        ax = df.plot(kind = 'bar',
                     figsize = (16,5),
                     color = 'red',
                     alpha = 0.75,
                     rot = 45,
                     fontsize = 13
                     )
        ax.set_title("F1 Acc. Scores DL İçin")
        for i in ax.patches:
            ax.annotate(str(round(i.get_height(), 3)),
                        (i.get_x() + 0.1, i.get_height() * 1.002), color='black', fontsize=8)


def run(algorithm):
    if(algorithm == 'ML'):
        plotBars(word_scores, "Word", 'ML')
        plotBars(char_scores, "Char", 'ML')
        plotBars(char_wb_scores, "Char_Wb", 'ML')
    elif(algorithm == 'DL'):
        plotBars(dl_scores,'None','DL')




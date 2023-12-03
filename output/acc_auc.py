import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

def avg_score(csv_read, csv_write):
    df = pd.read_csv(csv_read)
    df['img'] = [item.split('/')[-2].split('_')[0] for item in df['img']]
    grouped_df = df.groupby('img')['predict'].mean().reset_index()
    grouped_df.to_csv(csv_write, index=False)

def ACC(csv_f, csv_r, threshold):
    data_f = pd.read_csv(csv_f)
    data_r = pd.read_csv(csv_r)
    score_f = data_f['predict'].tolist()
    score_r = data_r['predict'].tolist()
    correct_f = sum(1 for score in score_f if score < threshold)
    correct_r = sum(1 for score in score_r if score >= threshold)
    acc = (correct_r + correct_f) / (len(score_r) + len(score_f))
    return acc

def AUC(csv_f, csv_r):
    data_f = pd.read_csv(csv_f)
    data_r = pd.read_csv(csv_r)
    score_f = np.array(data_f['predict'])
    score_r = np.array(data_r['predict'])
    y_true_f = np.zeros_like(score_f)
    y_true_r = np.ones_like(score_r)
    scores = np.concatenate((score_f, score_r))
    y_true = np.concatenate((y_true_f, y_true_r))
    auc = roc_auc_score(y_true, scores)
    return auc

# create avg score csv
csv_read_f = r'res_f.csv'
csv_read_r = r'res_r.csv'
csv_write_f = r'res_f_avg.csv'
csv_write_r = r'res_r_avg.csv'
avg_score(csv_read_f, csv_write_f)
avg_score(csv_read_r, csv_write_r)

def frange(start, end, step):
    current = start
    while current < end:
        yield current
        current += step


start = 0.40
end = 0.61
step = 0.01
decimal_sequence = [round(x, 2) for x in list(frange(start, end, step))]

accs1 = []
auc_result = 0
for i in decimal_sequence:
    threshold = i
    acc = ACC(csv_write_f, csv_write_r, threshold)
    accs1.append(acc)
    auc_result = AUC(csv_write_f, csv_write_r)
print('AUC: ', auc_result)
max_acc = max(accs1)
print('Acc: ', max_acc)
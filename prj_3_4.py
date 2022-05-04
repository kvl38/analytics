import pandas
import sklearn.metrics

#1
data = pandas.read_csv('classification.csv')

#2
tp = fp = fn = tn = 0

for i in range(0, len(data)):
    if data['true'][i] == 1 and data['pred'][i] == 1:
        tp += 1
    if data['true'][i] == 0 and data['pred'][i] == 0:
        tn += 1
    if data['true'][i] == 0 and data['pred'][i] == 1:
        fp += 1
    if data['true'][i] == 1 and data['pred'][i] == 0:
        fn += 1

# print("tp, fp, fn, tn = ", tp, fp, fn, tn)

#3
res1 = sklearn.metrics.accuracy_score(data['true'], data['pred'])
res2 = sklearn.metrics.precision_score(data['true'], data['pred'])
res3 = sklearn.metrics.recall_score(data['true'], data['pred'])
res4 = sklearn.metrics.f1_score(data['true'], data['pred'])
# print("{:.2f} {:.2f} {:.2f} {:.2f}".format(res1, res2, res3, res4))

#4
data = pandas.read_csv('scores.csv')

#5
scores = {}
for x in data.columns[1:]:
    scores[x] = round(sklearn.metrics.roc_auc_score(data['true'], data[x]), 2)

# print(scores)

#6
scores2 = {}
for x in data.columns[1:]:
    curve = sklearn.metrics.precision_recall_curve(data['true'], data[x])
    df = pandas.DataFrame({'precision': curve[0], 'recall': curve[1]})
    scores2[x] = round(df[df['recall'] >= 0.7]['precision'].max(), 2)

print(scores2)
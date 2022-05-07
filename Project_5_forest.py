import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

#1
data = pd.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

print(data.head())

#2
X = data.drop('Rings', axis=1)
y = data['Rings']


#3
kf = KFold(n_splits=5, shuffle=True, random_state=1)
results = []
for n in range(1, 51):
    tree = RandomForestRegressor(n_estimators=n, random_state=1)
    tree.fit(X, y)
    arr = cross_val_score(estimator=tree, X=X, y=y, cv=kf, scoring='r2')
    m = arr.mean()
    results.append(m)
for i in results:
    print(i)


#4
for index, i in enumerate(results):
    if i >= 0.52:
        print(i)
        print(index+2)
        break
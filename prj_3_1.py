import pandas as pd
from sklearn.svm import SVC

#1
data = pd.read_csv('svm-data.csv', header=None)
x = data[data.columns[1:3]]
y = data[data.columns[0]]

#2
clf = SVC(C=100000, random_state=241, kernel='linear')
print(clf.fit(x, y))

#3
num = clf.support_
for i in num:
    if i != num[len(num)-1]:
        print("".join(str(i + 1)) + ",", end=" ")
    else:
        print("".join(str(i + 1)), end=" ")



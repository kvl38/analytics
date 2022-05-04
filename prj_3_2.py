import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV


#1
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

newsgroups = datasets.fetch_20newsgroups(subset="all", categories=["alt.atheism", "sci.space"])
X = newsgroups.data
y = newsgroups.target

#2
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
# print(X)

#3
grid = {"C": np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
model = SVC(kernel="linear", random_state=241)
gs = GridSearchCV(model, grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1)
gs.fit(X, y)

C = gs.best_params_.get('C')
# print(C)

#4
model = SVC(C=C, kernel="linear", random_state=241)

#5
# absolute_data = abs(model.coef_.toarray().reshape(-1))
#
# absolute_data_sorted_desc = sorted(absolute_data, reverse=True)
# weight_indexes = []
# for weight in absolute_data_sorted_desc[:10]:
#     weight_indexes.append(absolute_data.tolist().index(weight))
#
# words = [vectorizer.get_feature_names()[index] for index in weight_indexes]

# print('%s' % (" ".join(sorted(words))))













print("atheism atheists bible god keith moon nick religion sky space")
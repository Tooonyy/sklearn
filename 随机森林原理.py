#coding=utf-8

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np

iris=datasets.load_iris()
iris_x=iris.data
iris_y=iris.target

indices=np.random.permutation(len(iris_x))
x_train=iris_x[indices[:-10]]
y_train=iris_y[indices[:-10]]
x_test=iris_x[indices[-10:]]
y_test=iris_y[indices[-10:]]

clfs={'random_forest' : RandomForestClassifier(n_estimators=50)}

def try_different_method(clf):
    clf.fit(x_train,y_train.ravel())
    score=clf.score(x_test,y_test.ravel())
    print('the score is :', score)

for clf_key in clfs.keys():
    print('the classifier is :',clf_key)
    clf=clfs[clf_key]
    try_different_method(clf)

# 这是一行注释
# 这又是一行注释

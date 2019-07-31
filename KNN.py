#-*- coding:utf-8 -*-
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris=datasets.load_iris()
iris_x=iris.data
iris_y=iris.target

indices=np.random.permutation(len(iris_x))
iris_x_train=iris_x[indices[:-10]]
iris_y_train=iris_y[indices[:-10]]

iris_x_test=iris_x[indices[-10:]]
iris_y_test=iris_y[indices[-10:]]

knn=KNeighborsClassifier()
knn.fit(iris_x_train,iris_y_train)
iris_y_predict=knn.predict(iris_x_test)
score=knn.score(iris_x_test,iris_y_test,sample_weight=None)

print('iris_y_predict= ')
print(iris_y_predict)
print('iris_y_test= ')
print(iris_y_test)

print 'Accuracy:', score


#this is a line of comments
#this is another line of comments
#third line of comments

from sklearn import svm

x=[[0,0],[2,2]]
y=[0.5,2.5]
clf=svm.SVR()
clf.fit(x,y)
result=clf.predict([[1,1]])
print result

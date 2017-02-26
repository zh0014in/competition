import numpy as np
import pandas as pd
from numpy.core.defchararray import index

train = pd.read_csv('train.csv',sep=",")
data = np.array(train)

#f = open("train.csv")
#f.readline()  # skip the header
#data = np.loadtxt(f, delimiter=",")

X = data[:, 3:]  # select columns 1 through end
y = data[:, 1]   # select column 0, the stock price

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(X,y)

#test = pd.read_csv('test.csv', sep=",")
#data = np.array(test)

result = clf.predict(data[:,3:])

thefile = open('result.csv', 'w')
thefile.write("Id,Prediction\n")
errorCount = 0
for index, item in enumerate(result):
    thefile.write("%s" % (index+1))
    thefile.write(",%s" % item)
    thefile.write(",%s\n" % y[index])
    if item != y[index]:
        errorCount = errorCount+1
        
thefile.write("%s, " % errorCount)
thefile.write("%s\n" % len(result))
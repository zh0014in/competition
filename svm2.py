import numpy as np
import pandas as pd
from numpy.core.defchararray import index

train = pd.read_csv('train.csv',sep=",")
data_train = np.array(train)

i_train = data_train[:, 0]
X_train = data_train[:, 3:]  
y_train = data_train[:, 1]   

print i_train
print y_train
p_train = np.column_stack([i_train, y_train])
print p_train

from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100.)
 
clf.fit(X_train,y_train)
#score = clf.score(X_train,y_train)
#print(score)

test = pd.read_csv('test.csv', sep=",")
data_test = np.array(test)
X_test = data_test[:, 3:]
i_test = data_test[:, 0]
result = clf.predict(data_train[:,3:])
print len(result)
print result

result = np.column_stack([i_test, result])
p_train = np.vstack([p_train, result])
print p_train
# a = np.concatenate((i_train,i_test))
# print a
# for index,value in enumerate(a):
#     print value
    
# print(i_train)
# a = np.concatenate((i_train, y_train),axis = 0)
# print(a)
  
#===============================================================================
# thefile = open('result.csv', 'w')
# thefile.write("Id,Prediction\n")
# errorCount = 0
# for index, item in enumerate(result):
#     thefile.write("%s" % (index+1))
#     thefile.write(",%s" % item)
#     thefile.write(",%s\n" % y_train[index])
#     if item != y_train[index]:
#         errorCount = errorCount+1
#           
# thefile.write("%s, " % errorCount)
# thefile.write("%s\n" % len(result))
#===============================================================================

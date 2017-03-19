import numpy as np
import pandas as pd
from numpy.core.defchararray import index

train = pd.read_csv('train.csv',sep=",")
data_train = np.array(train)

B= np.split(data_train, np.where(data_train[:, 3]== 1.)[0][1:])
# B = np.split(data_train, np.argwhere(data_train[:,3] == 1).flatten()[1:])
# print B
print B[0].shape[0]
label = []
A = []
for item in B:
    label = np.append(label, [''.join(item[:, 1])], axis=0)
#     A = np.append(A, item[:, 4:].flatten(), axis = 0)
print label
i_train = data_train[:, 0]
X_train = data_train[:, 2:]  
y_train = data_train[:, 1]

next_id = X_train[:, 0]
next_id[next_id > -1] = 1
X_train[:, 0] = next_id

#remove features with low variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X_train)
print X_train

p_train = np.column_stack([i_train, y_train])

from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100.)

print clf.fit(A,label)

test = pd.read_csv('test.csv', sep=",")
data_test = np.array(test)
X_test = data_test[:, 2:]
i_test = data_test[:, 0]

next_id = X_test[:, 0]
next_id[next_id > -1] = 1
X_test[:, 0] = next_id
#remove features with low variance
from sklearn.feature_selection import VarianceThreshold
sel.fit_transform(X_test)

target = pd.read_csv('target.csv', sep=",")
data_target = np.array(target)
y_test = data_target[:, 1]
 
score = clf.score(X_test,y_test)
print(score)

# result = clf.predict(X_test)
#  
# result = np.column_stack([i_test, result])
# result = np.vstack([p_train, result])
# print result
# thefile = open('result.csv', 'w')
# thefile.write("Id,Prediction\n")
# for index, item in enumerate(result):
#     thefile.write("%s" % item[0])
#     thefile.write(",%s\n" % item[1])

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

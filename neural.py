import numpy as np
import pandas as pd
from numpy.core.defchararray import index

train = pd.read_csv('train.csv',sep=",")
data_train = np.array(train)

i_train = data_train[:, 0]
X_train = data_train[:, 3:]  
y_train = data_train[:, 1]

# next_id = X_train[:, 0]
# next_id[next_id > -1] = 1
# X_train[:, 0] = next_id

#remove features with low variance
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(X_train)
print X_train

p_train = np.column_stack([i_train, y_train])

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)
 
clf.fit(X_train,y_train)

test = pd.read_csv('test.csv', sep=",")
data_test = np.array(test)
X_test = data_test[:, 3:]
i_test = data_test[:, 0]

# next_id = X_test[:, 0]
# next_id[next_id > -1] = 1
# X_test[:, 0] = next_id
#remove features with low variance
# from sklearn.feature_selection import VarianceThreshold
# sel.fit_transform(X_test)

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

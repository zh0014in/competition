import numpy as np
import pandas as pd
from numpy.core.defchararray import index

train = pd.read_csv('train.csv',sep=",")
data_train = np.array(train)

i_train = data_train[:, 0]
X_train = data_train[:, 2:]  
y_train = data_train[:, 1]

next_id = X_train[:, 0]
next_id[next_id > -1] = 1
next_id[next_id == -1] = 0
X_train[:, 0] = next_id
print X_train

p_train = np.column_stack([i_train, y_train])

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

h = .02  # step size in the mesh
n_neighbors = 15

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()

# test = pd.read_csv('test.csv', sep=",")
# data_test = np.array(test)
# X_test = data_test[:, 2:]
# i_test = data_test[:, 0]
# 
# next_id = X_test[:, 0]
# next_id[next_id > -1] = 1
# next_id[next_id == -1] = 1
# X_test[:, 0] = next_id
# 
# target = pd.read_csv('target.csv', sep=",")
# data_target = np.array(target)
# y_test = data_target[:, 1]
#  
# score = clf.score(X_test,y_test)
# print(score)

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

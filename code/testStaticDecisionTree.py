"""
This script tests StaticDecisionTree in staticDecisionTreeClasifier module
"""

# Authors: Taro Tezuka <tez@sings.jp>

from __future__ import division

import numpy as np

from staticDecisionTreeClassifier import StaticDecisionTreeClassifier

X_train = np.array([[3,2,5,1,1,1,3,2,5,4], [7,3,1,-2,6,5,-5,8,1,6], [2,5,1,1,1,3,2,5,6,7],])
y_train = ['W', 'R', 'N']
# y_train = np.array([6,1,5,2,3])
x_test = np.array([[5,5,2,3,5,5,6,9,1,9], [7,3,1,-2,6,5,-5,8,1,6]])

classifier = StaticDecisionTreeClassifier()

classifier.fit(X_train, y_train)

pred = classifier.predict(x_test)

print('pred = ' + str(pred))

classifier.showThresh()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn import metrics

# download and read mnist
mnist = fetch_mldata('MNIST original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target

# split data to train and test (for faster calculation, just use 1/10 data)

X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
# TODO:use logistic regression
# choose module
cls = LogisticRegression()
cls.fit(X_train, Y_train)
train_accuracy = cls.score(X_train, Y_train)
test_accuracy = cls.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


# TODO:use naive bayes
cls = BernoulliNB()
cls.fit(X_train, Y_train)
train_accuracy = cls.score(X_train, Y_train)
test_accuracy = cls.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


# TODO:use support vector machine
# choose module
cls = LinearSVC(random_state=0, tol=1e-4)    # default parameters
cls.fit(X_train, Y_train)
train_accuracy = cls.score(X_train, Y_train)
test_accuracy = cls.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


# TODO:use support vector machine
# choose module
cls = LinearSVC(random_state=0, tol=1e-5, C=0.01)    # adjust parameters
cls.fit(X_train, Y_train)
train_accuracy = cls.score(X_train, Y_train)
test_accuracy = cls.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
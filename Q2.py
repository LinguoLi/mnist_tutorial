from sklearn.datasets import fetch_openml


X, Y = fetch_openml('mnist_784', version=1, data_home='./scikit_learn_data', return_X_y=True)


X = X / 255.


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

from sklearn.naive_bayes import BernoulliNB

model=BernoulliNB()
model.fit(X_train,Y_train)

train_accuracy=model.score(X_train,Y_train)
test_accuracy=model.score(X_test,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


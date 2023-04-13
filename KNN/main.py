import numpy as np
from knn import KNN

train = np.genfromtxt('mnist_train_min.csv', delimiter=',', skip_header=1, dtype=np.int_)
test = np.genfromtxt('mnist_test_min.csv', delimiter=',', skip_header=1, dtype=np.int_)
y_train = train[:, 0]
X_train = train[:, 1:]
y_test = test[:, 0]
X_test = test[:, 1:]


clf = KNN(k = 10)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

acc = np.sum(prediction == y_test) / len(y_test)
print(acc)
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
import pickle
from sklearn.metrics import confusion_matrix


# preparing data
mnist = fetch_openml('mnist_784', version=1, parser='auto')

print((np.array(mnist.data.loc[69999]).reshape(28, 28) > 0).astype(int))
print(mnist.data)


# wrong way of splitting data into training and testing set
X, y = mnist["data"], mnist["target"].astype(np.uint8)
y1 = y.sort_values()
X1 = X.reindex(y.index)

X1_train, X1_test = X1[:56000], X1[56000:]
y1_train, y1_test = y1[:56000], y1[56000:]
print(y1_train)
print(y1_test)


# the right way
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# stochastic gradient descent classifier to identify 0
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)

print(sgd_clf.predict([mnist["data"].loc[0], mnist["data"].loc[1]]))
print((np.array(mnist.data.loc[0]).reshape(28, 28) > 0).astype(int))
print((np.array(mnist.data.loc[1]).reshape(28, 28) > 0).astype(int))


# accuracy
y_train_pred = sgd_clf.predict(X_train)
y_test_pred = sgd_clf.predict(X_test)

acc_train = sum(y_train_pred == y_train_0)/len(y_train_0)
acc_test = sum(y_test_pred == y_test_0)/len(y_test_0)

sgd_acc = [acc_train, acc_test]
print(sgd_acc)
with open('sgd_acc.pkl', 'wb') as file:
  pickle.dump(sgd_acc, file)


# cross validation accuracy
score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
print(score)

array = np.ndarray((3,), buffer=np.array(score))
print(array)
with open('sgd_cva.pkl', 'wb') as file:
  pickle.dump(array, file)


# SGD classifier for all numbers
sgd_m_clf = SGDClassifier(random_state=42, n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)

print(sgd_m_clf.predict([mnist["data"].loc[0], mnist["data"].loc[1]]))
print((np.array(mnist.data.loc[0]).reshape(28, 28) > 0).astype(int))
print((np.array(mnist.data.loc[1]).reshape(28, 28) > 0).astype(int))

# accuracy
print(cross_val_score(sgd_m_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1))

y_test_pred = cross_val_predict(sgd_m_clf, X_test, y_test, cv=3, n_jobs=-1)
conf_mx = confusion_matrix(y_test, y_test_pred)
print(conf_mx)


with open('sgd_cmx.pkl', 'wb') as file:
  pickle.dump(conf_mx, file)
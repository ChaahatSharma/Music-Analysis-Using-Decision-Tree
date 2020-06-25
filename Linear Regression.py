import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
dataset = pd.read_csv('output.csv')
dataset.head()

Xs = dataset.drop('target', axis=1)
y = dataset['target']
reg = LinearRegression()
reg.fit(Xs, y)
X = dataset.drop('target', axis=1)
y = dataset['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) from sklearn.metrics import classification_report,
confusion_matrix, accuracy_score

print(accuracy_score(y_test, y_pred.round()))

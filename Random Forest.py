import pandas as pd
import numpy as np
dataset = pd.read_csv('output.csv')
dataset.head()

X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, 10].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cutoff = 0.99

y_pred_classes = np.zeros_like(y_pred)
y_pred_classes[y_pred > cutoff] = 1
y_test_classes = np.zeros_like(y_pred)
y_test_classes[y_test_classes > cutoff] = 1
print(accuracy_score(y_test_classes, y_pred_classes))

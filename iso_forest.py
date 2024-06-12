import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import IsolationForest

df = pd.read_csv("creditcard.csv")

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3141)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = IsolationForest(random_state=3141, contamination='auto')
model.fit(X_train_scaled)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

y_pred_train = np.where(y_pred_train == -1, 1, 0)  
y_pred_test = np.where(y_pred_test == -1, 1, 0)

print(classification_report(y_test, y_pred_test))

plt.figure(figsize=(10, 6))

# Plot inliers (correctly classified)
plt.scatter(X_test[y_test == 0].iloc[:, 0], X_test[y_test == 0].iloc[:, 1], c='blue', label='Inliers')

# Plot outliers (incorrectly classified)
plt.scatter(X_test[y_test == 1].iloc[:, 0], X_test[y_test == 1].iloc[:, 1], c='red', label='Outliers')

plt.title("Classification of Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
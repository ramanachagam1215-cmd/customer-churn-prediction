import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = {
    "age": [25, 32, 45, 28, 52, 40, 35, 60, 23, 48],
    "tenure": [1, 3, 6, 2, 8, 5, 4, 10, 1, 7],
    "monthly_charges": [50, 65, 90, 55, 100, 80, 70, 110, 45, 95],
    "churn": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

print(df)
X = df[["age", "tenure","monthly_charges"]]
y = df["churn"]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train)
print(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("predicted:", y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
new_customer = pd.DataFrame([[30, 2, 60 ]],columns=["age", "tenure","monthly_charges"])

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("New customer is likely to churn")
else:
    print("New customer is unlikely to churn")
import pandas as pd

df = pd.read_csv("Iris.csv")
print("\nFirst five rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

df = df.drop("Id",axis=1)
x = df.drop("Species", axis=1)
y = df["Species"]

print("\nInput features:")
print(x.head())

print("\nTarget species:")
print(y.head())

print("\nInput shape:", x.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

print("\nModel training completed successfully!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=45,
    cmap="Blues"
)
plt.title("Iris Flower Classification - Confusion Matrix")
plt.tight_layout()
plt.show()

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=x.columns
)

prediction = model.predict(new_flower)
print("\nNew Flower Prediction:")
print("Predicted Species:", prediction[0])
# Iris Flower Classification 🌸

## Project Overview
This project uses Machine Learning to classify Iris flowers into three species based on their sepal and petal measurements.

It was developed as part of the **CodeAlpha Data Science Internship**.

## Dataset
The dataset contains 150 flower samples with the following features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

**Target variable:** Species
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Machine Learning Algorithm
**Logistic Regression** was used to train the classification model.

The dataset was divided into:
- Training set: 80% (120 samples)
- Testing set: 20% (30 samples)

## Model Performance
The model achieved **96.67% accuracy** on the test dataset.

Evaluation methods:
- Accuracy Score
- Classification Report
- Confusion Matrix

## Project Features
- Load and explore the Iris dataset.
- Prepare input features and target labels.
- Split data into training and testing sets.
- Train a Logistic Regression model.
- Evaluate model performance.
- Predict the species of a new flower.

## How to Run the Project

### 1. Install Python
Make sure Python is installed on your computer.

### 2. Install Required Libraries
Run this command in your terminal:

```bash
pip install pandas scikit-learn matplotlib
```

### 3. Download the Project Files
Keep these two files in the same folder:
- `Iris_Classification.py`
- `Iris.csv`

### 4. Run the Python Script

```bash
python Iris_Classification.py
```

The program will display dataset information, model accuracy, the classification report, the confusion matrix, and a new flower prediction.

## Example Prediction
For the flower measurements:

- Sepal Length: 5.1 cm
- Sepal Width: 3.5 cm
- Petal Length: 1.4 cm
- Petal Width: 0.2 cm

**Predicted Species:** `Iris-setosa`

## Author
Junaid Raza

**Internship:** CodeAlpha Data Science Internship

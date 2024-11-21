# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Preprocess Data: Read the dataset, drop unnecessary columns, and convert categorical variables into numerical codes using .astype('category') and .cat.codes.
2. Define Variables: Split the dataset into features (X) and target variable (Y), and initialize a random parameter vector theta.
3. Implement Functions: Define the sigmoid, loss, gradient_descent, and predict functions for logistic regression.
4. Train Model: Use gradient descent to optimize the parameters theta over a specified number of iterations.
5. Evaluate and Predict: Calculate accuracy of predictions on the training data, and demonstrate predictions with new sample data.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: JOTHILAKSHMI P
RegisterNumber:212223110017  
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis = 1)
Y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.3f}")
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
```

## Output:

## aaccuracy score:
![image](https://github.com/user-attachments/assets/1bf37881-b4f9-4502-ba91-61355469e6bc)

## confusion matrix:
![image](https://github.com/user-attachments/assets/5760f154-73b2-4f92-8883-4437087e1177)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

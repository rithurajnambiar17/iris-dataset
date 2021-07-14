#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

#Loading our data
iris = load_iris()

#Seperating our feature variable and converting target variable to binary form
x = iris["data"][:, 2:] #Petal Length & width
y = (iris["target"] == 2).astype(np.int) #1 for virginica and 0 for non-virginica

data = pd.DataFrame(x, columns = ["Petal Length", "Petal Width"])
data["Virginica"] = y

X = data.drop(columns = ["Virginica"])
y = data["Virginica"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, random_state = 42)

def logistic(X, y, alpha = 0.2, epoch = 2):
    X = np.array(X)
    y = np.array(y)
    b0_list, b1_list, b2_list = [], [], []
    global b0, b1, b2, pred
    b0, b1, b2 = 0, 0, 0
    for j in range(1, epoch+1):
        print(f"Epoch : {j}/{epoch}\n")
        for i in range(0, len(X)):
            pred = 1/(1+np.exp(-(b0+ b1 * x[i][0] + b2 * x[i][0])))
            b0 = b0 + alpha * (y[i] - pred) * pred * (1 - pred)
            b1 = b1 + alpha * (y[i] - pred) * pred * (1 - pred) * x[i][0]
            b2 = b2 + alpha * (y[i] - pred) * pred * (1 - pred) * x[i][1]
            b0_list.append(b0)
            b1_list.append(b1)
            b2_list.append(b2)
        print(f"Epoch {j} Completed")
        print(f"b0 : {b0}\nb1 : {b1}\nb2 : {b2}\n")

def predict(x, b0, b1, b2):
    global pred_list
    x = np.array(x)
    pred_list = []
    for i in range(0, len(x)):
        temp = 1/(1+np.exp(-(b0 + b1 * x[i][0] + b2 * x[i][1])))
        pred_list.append(temp)
    return pred_list

def crisp_logistic():
    for i in range(0, len(pred_list)):
        if pred_list[i] >= 0.5:
            pred_list[i] = 1
        elif pred_list[i] < 0.5:
            pred_list[i] = 0
    return np.array(pred_list)

def pred_score(y_true, y_pred):
    true = 0
    y_true = np.array(y_true)
    for i in range(0, len(y_true)):
        if y_pred[i] == y_true[i]:
            true+=1
        else:
            continue
    accuracy = (true/len(y_pred))*100
    return accuracy

logistic(X_train, y_train, epoch = 35)
predict(X_test, b0, b1, b2)
crisp_logistic()
pred_score(y_test, pred_list)
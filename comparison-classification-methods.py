import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

#Tahmin Kütüphaneleri
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

dataSet = pd.read_csv("Iris.csv")
del dataSet["Id"]

labelEncoder = preprocessing.LabelEncoder()

X = dataSet.iloc[:,:4].values
Y = dataSet.iloc[:,4:].values

Y = labelEncoder.fit_transform(Y)
standardScaler = StandardScaler()

x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size= 0.33 , random_state=0)

Succes_Rates = []

labels = ["Logistic Regression" , "KNN Algorithm" , "Support Vector Machine" , "Naive Bayes" , "Decision Tree" , "Rassal Forest" , "XGBoost" , "Deep Learning"   ]

def succesRateCalculate(matrix):
    
    truePred = matrix[0][0] + matrix[1][1] + matrix[2][2]
    falsePred = matrix[0][1] + matrix[1][0] + matrix[2][0] + + matrix[2][1] + + matrix[1][2] + + matrix[0][2]
    
    succesRate = (truePred / (truePred + falsePred)) * 100
    
    return succesRate


def confPlot(matrix , name):
    truePred = matrix[0][0] + matrix[1][1] + matrix[2][2]
    falsePred = matrix[0][1] + matrix[1][0] + matrix[2][0] + + matrix[2][1] + + matrix[1][2] + + matrix[0][2]
    
    plt.bar([0.25] , [truePred] , label = "True Predictions" , width=1 , color = "green")
    plt.bar([1.50] , [falsePred] , label = "False Predictions" , width=1 , color = "red")
    plt.legend()
    plt.ylabel('Prediction Results')
    title = name + " Predictions"
    plt.title(title)
    plt.show()
    
    
#Logistic Regression
logisticRegression = LogisticRegression(random_state = 0 )
logisticRegression.fit(x_train ,y_train)
logisticRegression_prediction = logisticRegression.predict(x_test)

logisticRegression_confmatrix = confusion_matrix(y_test , logisticRegression_prediction)

Succes_Rates.append(succesRateCalculate(logisticRegression_confmatrix).astype(int))

confPlot(logisticRegression_confmatrix , "Logistic Regression")


#KNN Algorithm
knn = KNeighborsClassifier(n_neighbors=5 , metric="minkowski")
knn.fit(x_train , y_train)
knn_predictions = knn.predict(x_test)

knn_confmatrix = confusion_matrix(knn_predictions , y_test)
Succes_Rates.append(succesRateCalculate(knn_confmatrix).astype(int))
confPlot(knn_confmatrix, "KNN")

#Support Vector Machine
svc = SVC(kernel = "linear")
x_train_ss = standardScaler.fit_transform(x_train)
x_test_ss = standardScaler.fit_transform(x_test)

svc.fit(x_train_ss , y_train)
svc_predictions = svc.predict(x_test)

svc_confmatrix = confusion_matrix(y_test , svc_predictions)
Succes_Rates.append(succesRateCalculate(svc_confmatrix).astype(int))
confPlot(svc_confmatrix , "SVC")

#Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train , y_train)
gnb_predictions = gnb.predict(x_test)

gnb_confmatrix = confusion_matrix(y_test , gnb_predictions)
Succes_Rates.append(succesRateCalculate(gnb_confmatrix).astype(int))
confPlot(gnb_confmatrix , "Naive Bayes")


#Decision Tree
decisionTree = DecisionTreeClassifier(criterion="entropy")
decisionTree.fit(x_train , y_train)
decisionTree_predictions = decisionTree.predict(x_test)

decisionTree_confmatrix = confusion_matrix(y_test , decisionTree_predictions)
Succes_Rates.append(succesRateCalculate(decisionTree_confmatrix).astype(int))
confPlot(decisionTree_confmatrix , "Decision Tree")


#Rassal Forest
randomForest = RandomForestClassifier(n_estimators=5)
randomForest.fit(x_train , y_train)
randomForest_predictions = randomForest.predict(x_test)

randomForest_confmatrix = confusion_matrix(y_test , randomForest_predictions)
Succes_Rates.append(succesRateCalculate(randomForest_confmatrix).astype(int))
confPlot(randomForest_confmatrix , "Random Forest")


#XGBoost
xgbClassifier = XGBClassifier()
xgbClassifier.fit(x_train , y_train)
xgb_predictions = xgbClassifier.predict(x_test)

xgb_confmatrix = confusion_matrix(y_test , xgb_predictions)
Succes_Rates.append(succesRateCalculate(xgb_confmatrix).astype(int))
confPlot(xgb_confmatrix , "XGBoost")


#Deep Learning 
from keras.models import Sequential
from keras.layers import Dense

EPOCHS = 10

classifier = Sequential()

classifier.add(Dense(25 , activation = "relu" , input_dim = 4 , name = "input-layer" ))
classifier.add(Dense(25 , activation = "relu" , name = "hidden-layer"))
classifier.add(Dense(1 , activation = "sigmoid" ,name = "output-layer"))

classifier.compile(optimizer = "adam", loss = "categorical_crossentropy" , metrics=['accuracy'])

classifier.fit(x_train_ss , y_train , epochs=EPOCHS)

dl_predictions = classifier.predict(x_test_ss)
dl_predictions = (dl_predictions > 0.5)

dl_confmatrix = confusion_matrix(y_test  , dl_predictions)
Succes_Rates.append(succesRateCalculate(dl_confmatrix).astype(int))
confPlot(dl_confmatrix , "Deep Learning")

plt.figure(figsize=(12,6))

plt.barh(labels,Succes_Rates , label="Rates",color="#f05131") 

plt.legend()
plt.xlabel("Percentage")
plt.title("Succes Rates of Algorithms")
plt.show()

































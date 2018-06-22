import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
#----------------------------------------------------------------
#Read the Training and Testing Data:
#----------------------------------------------------------------
data_test = pd.read_csv(filepath_or_buffer="poker-hand-training-true.data.txt", sep=',', header=None)
data_train = pd.read_csv(filepath_or_buffer="poker-hand-testing.data.txt", sep=',', header=None)
#data_train = data_test


#----------------------------------------------------------------
#Print it's Shape to get an idea of the data set:
#----------------------------------------------------------------
print(data_train.shape)
print(data_test.shape)
#----------------------------------------------------------------
#Prepare the Data for Training and Testing:
#----------------------------------------------------------------
#Lendo os dados de treinamento
array_train = data_train.values
data_train = array_train[:,0:10]
label_train = array_train[:,10]
#Lendo os dados de teste
array_test = data_test.values
data_test = array_test[:,0:10]
label_test = array_test[:,10]
#----------------------------------------------------------------
# Scaling the Data for our Main Model
#----------------------------------------------------------------


# Scale the Data to Make the NN easier to converge
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(data_train)  
# Transform the training and testing data
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
#----------------------------------------------------------------
# Init the Models for Comparision
#----------------------------------------------------------------

k = int(input("Qual o tamanho de n? "))

models = [KNeighborsClassifier(n_neighbors=k),GaussianNB(),tree.DecisionTreeClassifier(),
          svm.SVC(kernel='linear', C=1), OneVsRestClassifier(svm.SVC(kernel='linear'))]

model_names = ["KNN","Naive Bayes","Decision Tree", "SVM One VS One","SVM One VS All"]
#----------------------------------------------------------------
# Run Each Model
#----------------------------------------------------------------
for model,name in zip(models,model_names):
    model.fit(data_train, label_train) 
    #Predict
    prediction = model.predict(data_test)
    # Print Accuracy
    acc = accuracy_score(label_test, prediction)
    print("Accuracy Using",name,": " + str(acc)+'\n')
    #print(classification_report(label_test,prediction))
    #print(confusion_matrix(label_test, prediction))
    #print("\n\n########################################################################################################\n\n")




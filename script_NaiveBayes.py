import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import math
import matplotlib.pyplot as plt

with open('optdigits_training.csv') as trainingFile:
    reader = csv.reader(trainingFile)
    X = []
    Y = []

    for row in reader:
        X.append(row[:64])
        Y.append(row[64])

    print("1st thing",X)

# print("Done with Loading the Training Data.")


# In[3]:


length_TrainingSet = len(X)
percentage_training = 0.9
len_train = math.floor(length_TrainingSet * percentage_training);

X_train = X[:len_train]
Y_train = Y[:len_train]
# print("Done with forming the Training Dataset.")

for i in range(0, len(X_train)):
    ls = X_train[i]
    for j in range(0, len(ls)):
        ls[j] = int(ls[j])
    X_train[i] = ls
for i in range(0, len(Y_train)):
    Y_train[i] = int(Y_train[i])


X_validation = X[len_train:len(X)]
Y_validation = Y[len_train:len(Y)]

for i in range(0, len(X_validation)):
    ls = X_validation[i]
    for j in range(0, len(ls)):
        ls[j] = int(ls[j])
    X_validation[i] = ls
for i in range(0, len(Y_validation)):
    Y_validation[i] = int(Y_validation[i])
# print("Done with forming the  Validation Dataset.")


# In[3]:

clf = GaussianNB()
clf = clf.fit(X_train, Y_train)
print("Classification is Done.")

# output_Predicted = clf.predict(X_train);
# accuracy_training = metrics.accuracy_score(output_Predicted, Y_train)
# print("Accuracy on the Training Data set with k ")
# print(accuracy_training * 100)

output_predicted_validation = clf.predict(X_validation)
accuracy_2ndFold = metrics.accuracy_score(output_predicted_validation, Y_validation)
print("Accuracy on the Validation Data set is:")
print(accuracy_2ndFold * 100)



## This code is responsible for formation of the Testing dataset.
with open('optdigits_test.csv') as testingFile:
    reader = csv.reader(testingFile)

    X_test = []
    Y_test = []

    for row in reader:
        X_test.append(row[:64])
        Y_test.append(row[64])


for i in range(0, len(X_test)):
    ls = X_test[i]
    for j in range(0, len(ls)):
        ls[j] = float(int(ls[j]))
    X_test[i] = ls
for j in range(0, len(Y_test)):
    Y_test[j] = float(int(Y_test[j]))

print("Done forming the Testing Dataset.")

### Prediction of the
output_predicted_testing = clf.predict(X_test)
accuracy_testing = metrics.accuracy_score(output_predicted_testing, Y_test)
print("Accuracy on the Testing Dataset is : ")
print(accuracy_testing * 100)


# In[ ]:




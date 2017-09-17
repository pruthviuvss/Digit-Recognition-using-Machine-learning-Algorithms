#loading the required libraries for tree construction and visulaisation.
library(rpart)
library(rattle)

#setting the working directory to the path containing datasets.
setwd("D:/pruthvi/Spring-17/ML/ITCS6156_SLProject/DigitRecognition")

#assigning train and test data to variables with a division of dependant class label
# and the decision attributes into separate variables.
train_data <- read.csv("optdigits_training.csv",header = FALSE)[,1:64]
class_label<-read.csv("optdigits_training.csv",header = FALSE)[,65]
test_data<-read.csv("optdigits_test.csv",header = FALSE)[,1:64]
class_label_test<-read.csv("optdigits_test.csv",header = FALSE)[,65]


#Exploring the dataset.
head(train_data,5)
nrow(train_data)
ncol(train_data)
str(train_data)

col_names<-colnames(train_data)

#creating a training model using rpart. 
fit <- rpart(class_label~.,train_data,
             method="class",control = rpart.control(cp=0.00001))

#predicting the values using the model and test dataset
predict<-predict(fit,test_data,type="class")

#cost matrix for the predictions
cm <- table(predict,class_label_test)

#calculating the accuracy from cost matrix
diag<-diag(cm)
n<-sum(cm)
acc<-sum(diag)/n
print(acc*100)


# pruning the above model to check the accuracy.
pfit<- prune(fit, cp= fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
prune_predict<-predict(pfit,test_data,type="class")
cm_prune<-table(prune_predict,class_label_test)
print(cm_prune)
diag_prune<-diag(cm_prune)
n_prune<-sum(cm_prune)
acc_prune<-sum(diag_prune)/n_prune
print(acc_prune*100)

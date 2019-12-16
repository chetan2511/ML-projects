#Machine Learning assignment on cars dataset

#Loading libraries
library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(ggord)
library(ggplot2)
library(Hmisc)
library(klaR)
library(MASS)
library(nnet)
library(plyr)
library(pROC)
library(psych)
library(scatterplot3d)
library(SDMTools)
library(dplyr)
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)
library(caret)
library(car)
library(DMwR)
library(rattle)

cars <- read.csv("Cars.csv")
head(cars)
View(cars)
attach(cars)  
summary(Transport)
#to check any  missing values
nrow(cars[is.na(cars),])
str(cars)

cars$CarUsage=ifelse(cars$Transport=='Car',1,0)
table(cars$CarUsage)
sum(cars$CarUsage)/nrow(cars)


cars$CarUsage=as.factor(cars$CarUsage)


hist(Work.Exp)
hist(Age)
hist(Salary)
#displays that the major portion of the group falls in the bracket of 22-30 yrs
library(DataExplorer)
plot_correlation(cars)
#displays that age, salary and work experience are related and we are excluding 
#variable work exp  from the dataset

str(cars)
cars$Age<-as.numeric(cars$Age)
cars$Gender<-as.numeric(cars$Gender)
cars$Engineer<-as.numeric(cars$Engineer)
cars$MBA<-as.numeric(cars$MBA)
cars$Work.Exp<-as.numeric(cars$Work.Exp)
cars$license<-as.numeric(cars$license)
cars$license<-as.numeric(cars$license)
cars$CarUsage=as.numeric(cars$CarUsage)
str(cars)
OLS.full<-lm(CarUsage~Age+Gender+Engineer+MBA+Work.Exp+Salary+Distance+
               license, data=cars)
summary(OLS.full)
vif(OLS.full)

#removing work exp from the dataset as its showing multicollinearity 
#with age and salary using corrplot

OLS.full1<-lm(CarUsage~Age+Gender+Engineer+MBA+Salary+Distance+
                license, data=cars)
summary(OLS.full1)
vif(OLS.full1)

pred.reg<-predict(OLS.full1,newdata=cars, interval="predict")
pred.reg
View(pred.reg)

cars1=cars[,c(1:4,6:10)]
str(cars1)


#model building and data split

str(cars1)
#convert to a data frame
cars2<-as.data.frame(cars1)
cars2$CarUsage=cars2$CarUsage -1

cars2$CarUsage<-as.factor(cars2$CarUsage)
cars2<-cars2[,c(9,1,2,3,4,5,6,7)]
str(cars2)

set.seed(123)


#splitting the data based on carusage

carindex<-createDataPartition(cars2$CarUsage, p=0.7,list = FALSE,times = 1)

carsdatatrain<-cars2[carindex,]
carsdatatest<-cars2[-carindex,]

prop.table(table(carsdatatrain$CarUsage))
prop.table(table(carsdatatest$CarUsage))

View(carsdatatrain)
carsdatatrain<-carsdatatrain[,c(1:8)]
carsdatatest<-carsdatatest[,c(1:8)]
## The train and test data have almost same percentage of cars usage as the base data

#we can check the ratios of the minority class to majority class
table(carsdatatrain$CarUsage)
table(carsdatatest$CarUsage)

#build model
#logistic regression
cars_logistic <- glm(CarUsage~., data=carsdatatrain, family=binomial(link="logit"))
carsdatatest$log.pred<-predict(cars_logistic, carsdatatest[1:8], type="response")
table(carsdatatest$CarUsage,carsdatatest$log.pred>0.5)

###predict the response of the model using the train data

predTrain=predict(cars_logistic,newdata=carsdatatrain,type='response')

#plot the ROC curve for calculating AUC
library(ROCR)
ROCRpred=prediction(predTrain,carsdatatrain$CarUsage)
as.numeric(performance(ROCRpred,'auc')@y.values)
perf_train=performance(ROCRpred,'tpr','fpr')
plot(perf_train,col='black',lty=2,lwd=2)
plot(perf_train,lwd=3,colorize=TRUE)

ks_train <- max(perf_train@y.values[[1]]- perf_train@x.values[[1]])
plot(perf_train,main=paste0('KS=',round(ks_train*100,1),'%'))
lines(x = c(0,1),y=c(0,1))

###predict the response of the model using the test data
 
predTest=predict(cars_logistic,newdata=carsdatatest,type='response')
#build confusion matrix; >0.5=true else false
conf_mat=table(carsdatatest$CarUsage,predTest>0.5)
conf_mat
#get accuracy by using the right classifiers
(conf_mat[1,1]+conf_mat[2,2])/nrow(na.omit(test))

#plot the ROC curve for calculating AUC
library(ROCR)
ROCRpred=prediction(predTest,carsdatatest$CarUsage)
as.numeric(performance(ROCRpred,'auc')@y.values)
perf_test=performance(ROCRpred,'tpr','fpr')
plot(perf_test,col='black',lty=2,lwd=2)
plot(perf_test,lwd=3,colorize=TRUE)

ks_test <- max(perf_test@y.values[[1]]- perf_test@x.values[[1]])
plot(perf_test,main=paste0('KS=',round(ks_test*100,1),'%'))
lines(x = c(0,1),y=c(0,1))

#knn
str(cars2)
library(class)

#convert variables to num in order to use knn
tcnorm<-scale(cars2[,-1])
tcnorm<-cbind(cars2[,1],tcnorm)
colnames(tcnorm)[1]<-'CarUsage'
View(tcnorm)

#convert to a data frame

df_tcnorm<-as.data.frame(tcnorm)
df_tcnorm$CarUsage<-as.factor(df_tcnorm$CarUsage)

#check number values 
table(df_tcnorm$CarUsage)

#partition the data
library(caTools)


#train using knn

sqrt(nrow(carsdatatrain))
View(carsdatatrain)
knn_fit<- knn(carsdatatrain[,2:7], carsdatatest[,2:7], 
              cl= carsdatatrain[,1],k = 17,prob=TRUE) 

#check confusion matrix
table.knn=table(carsdatatest[,1],knn_fit)
table.knn

#check accuracy
sum(diag(table.knn)/sum(table.knn))

#ch loss
loss.knn<-table.knn[2,1]/(table.knn[2,1]+table.knn[1,1])
loss.knn
opp.loss.knn<-table.knn[1,2]/(table.knn[1,2]+table.knn[2,2])
opp.loss.knn
tot.loss.knn<-0.95*loss.knn+0.05*opp.loss.knn
tot.loss.knn
##################

##Naive Bayes

library(e1071)

nb_gd<-naiveBayes(x=carsdatatrain[,2:8], y=as.factor(carsdatatrain[,1]))

pred_nb<-predict(nb_gd,newdata = carsdatatest[,2:8])

table(carsdatatest[,1],pred_nb)

####
library(gbm)          # basic implementation using AdaBoost
library(xgboost)      # a faster implementation of a gbm
library(caret)  # an aggregator package for performing many machine learning models

## Bagging
library(ipred)
library(rpart)
cars.bagging <- bagging(CarUsage ~.,
                          data=carsdatatrain,
                          control=rpart.control(maxdepth=6, minsplit=20))


carsdatatest$pred.class <- predict(cars.bagging, carsdatatest)
str(carsdatatest)
###since pred class is in factor hence converting it to numeric to determine confusion matrix
carsdatatest$pred.class=as.numeric(as.character(carsdatatest$pred.clas))
carsdatatest$pred.class<- ifelse(carsdatatest$pred.class>0.5,1,0)


confusionMatrix(data=factor(carsdatatest$pred.class),
                reference=factor(carsdatatest$CarUsage),
                positive='1')
###
table(carsdatatest$CarUsage,carsdatatest$pred.class>0.5) 
##observerved 8 true values for cars


##trying with a different value for maxdepth and minsplit
cars.bagging <- bagging(CarUsage ~.,
                        data=carsdatatrain,
                        control=rpart.control(maxdepth=5, minsplit=15))
carsdatatest$pred.class <- predict(cars.bagging, carsdatatest)
str(carsdatatest)
###since pred class is in factor hence converting it to numeric to determine confusion matrix
carsdatatest$pred.class=as.numeric(as.character(carsdatatest$pred.clas))
carsdatatest$pred.class<- ifelse(carsdatatest$pred.class>0.5,1,0)


confusionMatrix(data=factor(carsdatatest$pred.class),
                reference=factor(carsdatatest$CarUsage),
                positive='1')
###
table(carsdatatest$CarUsage,carsdatatest$pred.class>0.5) 

##bagging with max depth=6;minsplit=20 is better

#Boosting

#trying some general boosting techniques.


boostcontrol <- trainControl(number=10)

xgbGrid <- expand.grid(
  eta = 0.3,
  max_depth = 1,
  nrounds = 50,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1, subsample = 1
)

carsxgb <-  train(CarUsage ~ .,carsdatatrain,trControl = boostcontrol,tuneGrid = xgbGrid,metric = "Accuracy",method = "xgbTree")
carsxgb$finalModel
##predict using test dataset
predictions_xgb=predict(carsxgb,carsdatatest)
confusionMatrix(predictions_xgb,carsdatatest$CarUsage)


#####XGB BOOST###
# XGBoost works with matrices that contain all numeric variables

View(carsdatatrain)
str(carsdatatrain)
cars_features_train<-as.matrix(carsdatatrain[,2:8])
cars_label_train<-as.matrix(carsdatatrain[,1])
cars_features_test<-as.matrix(carsdatatest[,2:8])

xgb.fit <- xgboost(
  data = cars_features_train,
  label = cars_label_train,
  eta = 0.001,
  max_depth = 3,
  min_child_weight = 3,
  nrounds = 10000,
  nfold = 5,
  objective = "binary:logistic",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)


carsdatatest$xgb.pred.class <- predict(xgb.fit, cars_features_test)

table(carsdatatest$CarUsage,carsdatatest$xgb.pred.class>0.5)

#or simply the total correct of the minority class
sum(carsdatatest$CarUsage==1 & carsdatatest$xgb.pred.class>=0.5)


#adjusting lr,md and nr

#adjusting nr

tp_xgb<-vector()
lr <- c(0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1)
md<-c(1,3,5,7,9,15)
nr<-c(2, 50, 100, 1000, 10000)
for (i in nr) {
  
  xgb.fit <- xgboost(
    data = cars_features_train,
    label = cars_label_train,
    eta = 0.7,
    max_depth = 5,
    nrounds = i,
    nfold = 5,
    objective = "binary:logistic",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  carsdatatest$xgb.pred.class <- predict(xgb.fit, cars_features_test)
  
  tp_xgb<-cbind(tp_xgb,sum(carsdatatest$CarUsage==1 & carsdatatest$xgb.pred.class>=0.5))
  
}

tp_xgb

#Stopping. Best iteration after 13 rounds at nr=0.003401

#adjusting lr or eta

tp_xgb<-vector()
lr <- c(0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1)
md<-c(1,3,5,7,9,15)
nr<-c(2, 50, 100, 1000, 10000)
for (i in lr) {
  
  xgb.fit <- xgboost(
    data = cars_features_train,
    label = cars_label_train,
    eta = i,
    max_depth = 5,
    nrounds = 50,
    nfold = 5,
    objective = "binary:logistic",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  carsdatatest$xgb.pred.class <- predict(xgb.fit, cars_features_test)
  
  tp_xgb<-cbind(tp_xgb,sum(carsdatatest$CarUsage==1 & carsdatatest$xgb.pred.class>=0.5))
  
}

tp_xgb

#adjusting md

tp_xgb<-vector()
lr <- c(0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1)
md<-c(1,3,5,7,9,15)
nr<-c(2, 50, 100, 1000, 10000)
for (i in md) {
  
  xgb.fit <- xgboost(
    data = cars_features_train,
    label = cars_label_train,
    eta = 0.7,
    max_depth = i,
    nrounds = 50,
    nfold = 5,
    objective = "binary:logistic",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  carsdatatest$xgb.pred.class <- predict(xgb.fit, cars_features_test)
  
  tp_xgb<-cbind(tp_xgb,sum(carsdatatest$CarUsage==1 & carsdatatest$xgb.pred.class>=0.5))
  
}

tp_xgb

#now we put them all into our best fit!

xgb.fit <- xgboost(
  data = cars_features_train,
  label = cars_label_train,
  eta = 0.7,
  max_depth = 5,
  nrounds = 50,
  nfold = 5,
  objective = "binary:logistic",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)

carsdatatest$xgb.pred.class <- predict(xgb.fit, cars_features_test)

sum(carsdatatest$CarUsage==1 & carsdatatest$xgb.pred.class>=0.5)
table(carsdatatest$CarUsage,carsdatatest$xgb.pred.class>=0.5)
##80% correct values observed for test dataset

#working with SMOTE
library(DMwR)

set.seed(123)
carindex<-createDataPartition(cars2$CarUsage, p=0.7,list = FALSE,times = 1)
carsdatatrain<-cars2[carindex,]
carsdatatest<-cars2[-carindex,]
prop.table(table(carsdatatrain$CarUsage))
attach(carsdatatrain)
carsdataSMOTE<-SMOTE(CarUsage~., carsdatatrain,  perc.over = 250,perc.under = 100)
prop.table(table(carsdataSMOTE$CarUsage))

###going with equal ratio in car usage

#model building using xgboost
#now put our SMOTE data into our best xgboost

smote_features_train<-as.matrix(carsdataSMOTE[,2:8])
smote_label_train<-as.matrix(carsdataSMOTE$CarUsage)

smote.xgb.fit <- xgboost(
  data = smote_features_train,
  label = smote_label_train,
  eta = 0.7,
  max_depth = 5,
  nrounds = 50,
  nfold = 5,
  objective = "binary:logistic",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)

smote_features_test<-as.matrix(carsdatatest[,2:8])
carsdatatest$smote.pred.class <- predict(smote.xgb.fit, smote_features_test)

table(carsdatatest$CarUsage,carsdatatest$smote.pred.class>=0.5)
##we get all the 10 cars as true

#Let us proceed with building the models
## Model Building We will use the Logistic regression method a model 
#on the SMOTE data to understand the factors influencing car usage.

outcomevar<-'CarUsage'
regressors<-c("Age","Salary","Distance","license","Engineer","MBA","Gender")
trainctrl<-trainControl(method = 'repeatedcv',number = 10,repeats = 3)
carsglm<-train(carsdataSMOTE[,regressors],carsdataSMOTE[,outcomevar],method = "glm", family = "binomial",trControl = trainctrl)
summary(carsglm$finalModel)

varImp(object = carsglm)

plot(varImp(object = carsglm), main="Vairable Importance for Logistic Regression")
#we see that distance and engineer are most significant.
carusageprediction<-predict.train(object = carsglm,carsdatatest[,regressors],type = "raw")
confusionMatrix(carusageprediction,carsdatatest[,outcomevar], positive='1')
#96.77%overall accuracy and 90% accuracy in predicting car 

##RF MODEL
rftrcontrol<-control <- trainControl(method="repeatedcv", number=10, repeats=3)
mtry<-sqrt(ncol(carsdatatrain))
tunegridrf <- expand.grid(.mtry=mtry)
carsrf<-train(CarUsage ~.,carsdatatrain,method = "rf", trControl=rftrcontrol, tuneGrid = tunegridrf)
carsrf$finalModel
plot(varImp(object=carsrf), main = "Variable Importance for Random Forest")
#OOB estimate of  error rate is  1.7% in training dataset. salary and age are most significant

#attempt on test data
predictions_rf<-predict(carsrf,carsdatatest)
confusionMatrix(predictions_rf,carsdatatest$CarUsage)
#98.38% overall accuracy; 80% accuracy for predicting cars

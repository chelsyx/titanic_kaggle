# Read in data
library(data.table)
data_train <- fread("data/train.csv")
data_test <- fread("data/test.csv")

# dim 891 12
# Char: Sex, Cabin, Embarked,,,,,Name,Ticket

## Missing:
## 177 miss age
## 2 miss embarked



############## EXPLORE ########################

data_train1 <- data_train

####### Check Connection vs sur

data_train1$ticket_dup <- duplicated(data_train1$Ticket, fromLast=T)|duplicated(data_train1$Ticket)
hist(table(data_train1$Ticket[data_train1$ticket_dup])) # most cases: 2 people have same ticket number;max 7

data_train1$relative <- data_train1$SibSp+data_train1$Parch
table(data_train1$ticket_dup, data_train1$relative!=0) # 75 buy ticket with others, but not relatives

data_train1$tckPal <- rep(0, nrow(data_train1))
for (tckNum in unique(data_train1$Ticket[data_train1$ticket_dup])){
	xxpal.mask <- data_train1$Ticket == tckNum
	data_train1$tckPal[xxpal.mask] <- sum(xxpal.mask)-1
}
hist(data_train1$tckPal)

table(data_train1$Survived, data_train1$tckPal) 
hist(data_train1$tckPal, col=rgb(1,0,0,0.5), main='Overlapping Histogram', xlab='Variable')
hist(data_train1$tckPal[data_train1$Survived==1], col=rgb(0,0,1,0.5), add=T)
box()
# 1<pal<3, more likely to survived


table(data_train1$Survived, data_train1$relative) 
hist(data_train1$relative, col=rgb(1,0,0,0.5), xlim=c(-1,11), breaks=-1:11,main='Overlapping Histogram', xlab='Variable')
hist(data_train1$relative[data_train1$Survived==1], xlim=c(-1,11), breaks=-1:11,col=rgb(0,0,1,0.5), add=T)
box()
# 1<relative<3, more likely to survived

cor(as.matrix(data_train1[,.(Survived,relative,tckPal,SibSp,Parch,ticket_dup)]))
#relative, tckPal, SibSp, Parch highly correlated; ticket_dup best in predict; keep SibSp, Parch, tckPal



####### Check Social-Economic Status vs sur

table(data_train1$Cabin=="", data_train1$Survived) #people in cabin are more likely to survived
data_train1$wCabin <- as.numeric(data_train1$Cabin!="")
data_train1$cabinL <- substr(data_train1$Cabin,0,1)
table(data_train1$cabinL, data_train1$Survived) #certain cabin letter are more likely to survived

hist(data_train1$Fare) #>500 outlier

aggregate(data_train1$Pclass, by=list(data_train1$cabinL),mean) #TABC 1class; G 3class; other mix
aggregate(data_train1$Fare, by=list(data_train1$cabinL),median) #T and A are first clss, but cheap ticket

table(data_train1$Pclass, data_train1$Survived) #1st class more likely to survived
boxplot(data_train1$Survived, data_train1$Fare) #high fare more likely to survived, but large CI



####### Check Demo Status vs sur

data_train1$female <- as.numeric(data_train1$Sex == "female")
table(data_train1$Sex,data_train1$Survived) ## female are more likely to survived

hist(data_train1$Age, col=rgb(1,0,0,0.5)) #20-30 most
hist(data_train1$Age[data_train1$Survived==1], col=rgb(0,0,1,0.5), add=T)
box()
# kids are more likely to survived

table(data_train1$Survived,data_train1$Embarked) # cherbourg more likely


##########################################################################

##### Preprocess

data_train$ticket_dup <- as.numeric(duplicated(data_train$Ticket, fromLast=T)|duplicated(data_train$Ticket))
data_train$relative <- data_train$SibSp+data_train$Parch
data_train$tckPal <- rep(0, nrow(data_train))
for (tckNum in unique(data_train$Ticket[data_train$ticket_dup])){
	xxpal.mask <- data_train$Ticket == tckNum
	data_train$tckPal[xxpal.mask] <- sum(xxpal.mask)-1
}
data_train$wCabin <- as.numeric(data_train$Cabin!="")
data_train$cabinL <- substr(data_train$Cabin,0,1)
data_train$female <- as.numeric(data_train$Sex == "female")

data_train$cabinL<-as.factor(data_train$cabinL)
is.na(data_train$Embarked) <- data_train$Embarked==""
data_train$Embarked <- as.factor(data_train$Embarked)


## Preprocess test set

data_test$ticket_dup <- as.numeric(duplicated(data_test$Ticket, fromLast=T)|duplicated(data_test$Ticket))
data_test$relative <- data_test$SibSp+data_test$Parch
data_test$tckPal <- rep(0, nrow(data_test))
for (tckNum in unique(data_test$Ticket[data_test$ticket_dup])){
	xxpal.mask <- data_test$Ticket == tckNum
	data_test$tckPal[xxpal.mask] <- sum(xxpal.mask)-1
}
data_test$wCabin <- as.numeric(data_test$Cabin!="")
data_test$cabinL <- substr(data_test$Cabin,0,1)
data_test$female <- as.numeric(data_test$Sex == "female")

data_test$cabinL<-as.factor(data_test$cabinL)
is.na(data_test$Embarked) <- data_test$Embarked==""
data_test$Embarked <- as.factor(data_test$Embarked)


##########################################################################

##### Logistc Regression with regularizer

library(dummies)
library(LiblineaR)

xmask.comp <- complete.cases(data_train)

pred_tr <- data_train[xmask.comp,.(Pclass,Fare,wCabin,SibSp,Parch,ticket_dup,female,Age)]
pred_tr <- data.matrix(pred_tr)
pred_tr <- cbind(pred_tr, dummy(data_train$Embarked[xmask.comp]))
target <- as.factor(data_train[xmask.comp, Survived])


tryCosts=c(200,150,100,50,30,1)
bestCost=NA
bestAcc=0
for(co in tryCosts){
    acc=LiblineaR(data=pred_tr,target=target,type=0,cost=co,bias = F, wi = c("0"=0.5955, "1"=0.4045), cross = 100)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
    bestCost=co
    bestAcc=acc
    }
    }

logit_fit <- LiblineaR(pred_tr, target, type = 0, cost = 100, bias = F, wi = c("0"=0.5955, "1"=0.4045), cross = 712)
## from W(weight), most infuential predictors: Pclass, gender,Embarked
## base line: 0.8033708 (leave-one-out cv)
logit_fit <- LiblineaR(pred_tr, target, type = 0, cost = 100, bias = F, wi = c("0"=0.5955, "1"=0.4045), cross = 0)


# Generate predictive result

#xmask.comp <- complete.cases(data_test)
pred_test <- data_test[,.(Pclass,Fare,wCabin,SibSp,Parch,ticket_dup,female,Age)]
pred_test <- data.matrix(pred_test)
pred_test <- cbind(pred_test, dummy(data_test$Embarked))

## Impute missing Age and Fare
library(mice)
tempData <- mice(pred_test,m=5,maxit=50,meth='pmm',seed=500); summary(tempData)
test_comp <- complete(tempData,1)

result_test <- predict(logit_fit,test_comp,proba=T,decisionValues=TRUE)

# Output table
output_tb <- cbind(data_test[,PassengerId], as.numeric(result_test$predictions)-1)
colnames(output_tb) <- c("PassengerId", "Survived") 
# 0.77033 accuracy on leader board
# same result as logit regression with less predictors 
# http://nbviewer.jupyter.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb

write.csv(output_tb, file="logit_pred.csv", row.names = F)

## change probability cutoff to see if logit reg get a better result??????

##########################################################################

#### Random Forest

library(randomForest)
library(missForest)

### Impute missing value for trainning set using missForest
data_train.imp <- missForest(data_train[,.(Survived,Pclass,Fare,wCabin,SibSp,Parch,ticket_dup,female,Age,Embarked,tckPal)], parallelize = "no", verbose = TRUE)
data_train.imp$OOBerror # 0.1906676
train_imp <- data_train.imp$ximp
train_imp$Survived<-as.factor(train_imp$Survived)

### run model
rf1<-randomForest(Survived~.,data=train_imp,importance=TRUE)
print(rf1)
# OOB estimate of  error rate: 16.61%

# importance of predictor
varImpPlot(rf1, main='') ## Importance: gender, age, pclass, fare, tckpal
impmx <- importance(rf1)


### predict using test set

#impute test set without target, using training set together

useImp <- data_train[,.(Pclass,Fare,wCabin,SibSp,Parch,ticket_dup,female,Age,Embarked,tckPal)]
useImp <- rbind(useImp, data_test[,.(Pclass,Fare,wCabin,SibSp,Parch,ticket_dup,female,Age,Embarked,tckPal)])
data_test.imp <- missForest(useImp, parallelize = "no", verbose = TRUE)
data_test.imp$OOBerror # 0.5215207
test_imp <- data_test.imp$ximp[(nrow(data_train)+1):nrow(data_test.imp$ximp),]

rf.pred1 <- predict( rf1, test_imp)   

# Output table
output_tb <- cbind(data_test[,PassengerId], as.numeric(rf.pred1)-1)
colnames(output_tb) <- c("PassengerId", "Survived") # 0.75120 accuracy, worse than logistic and gender based model
# worse than rf in kaggle tutorial; may bc feature eng: family size, age*pclass

write.csv(output_tb, file="rf1_pred.csv", row.names = F)




##########################################################################

## Know that a score of 0.79 - 0.81 is doing well on this challenge, and 0.81-0.82 is really going beyond the basic models! The dataset here is smaller than normal, so there is less signal to tap your models into.


# Revisit your assumptions about how you cleaned and filled the data.
# Be creative with additional feature engineering
# Experiment with different parameters for your random forest.
# Consider a different model approach

# Read in data
library(data.table)
data_train <- fread("data/train.csv")
data_test <- fread("data/test.csv")

##### Preprocess

## For imputation, stack predictors in training and test together

allPred <- rbindlist(list(data_train,data_test),use.names=TRUE,fill=TRUE,idcol = "Source")
ol.mask <- allPred$Fare==0|allPred$Fare>500
allPred$Fare[ol.mask]<-NA

# Impute Fare NAs by mean per pclass
fna.mask <- is.na(allPred$Fare)
aggregate(allPred$Fare[!fna.mask], by=list(allPred$Pclass[!fna.mask]), mean)
allPred$Fare[fna.mask&allPred$Pclass==1]<-84.02592
allPred$Fare[fna.mask&allPred$Pclass==2]<-21.64811
allPred$Fare[fna.mask&allPred$Pclass==3]<-13.37847

# Impute Age NAs by median
ana.mask <- is.na(allPred$Age)
allPred$Age[ana.mask]<-median(allPred$Age[!ana.mask])

# Create New features
allPred$relative <- allPred$SibSp+allPred$Parch

allPred$ticket_dup <- as.numeric(duplicated(allPred$Ticket, fromLast=T)|duplicated(allPred$Ticket))
allPred$tckPal <- rep(0, nrow(allPred))
for (tckNum in unique(allPred$Ticket[allPred$ticket_dup==1])){
	xxpal.mask <- allPred$Ticket == tckNum
	allPred$tckPal[xxpal.mask] <- sum(xxpal.mask)-1
}

allPred$wCabin <- as.numeric(allPred$Cabin!="")
allPred$cabinL <- substr(allPred$Cabin,0,1)
allPred$cabinL[allPred$cabinL==""]<-"Unknown"
allPred$cabinL<-as.factor(allPred$cabinL)

# Transform
allPred$female <- as.numeric(allPred$Sex == "female")

allPred$Embarked[allPred$Embarked==""] <- "S" ##impute by mode
allPred$Embarked <- as.factor(allPred$Embarked)

# Recover traing and test set
dataTr <- allPred[Source==1,]
dataTe <- allPred[Source==2,]
dataTe <- dataTe[,Survived:=NULL]


##########################################################################

##### Logistc Regression with regularizer

library(dummies)
library(LiblineaR)

pred_tr <- dataTr[,.(Pclass,Fare,wCabin,SibSp,Parch,relative,ticket_dup,tckPal,female,Age)]
pred_tr <- data.matrix(pred_tr)
pred_tr <- cbind(pred_tr, dummy(dataTr$Embarked),dummy(dataTr$cabinL))
target <- as.factor(dataTr[, Survived])


tryCosts=c(200,150,100,50,30,1)
bestCost=NA
bestAcc=0
for(co in tryCosts){
    acc=LiblineaR(data=pred_tr,target=target,type=0,cost=co,bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 100)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
    bestCost=co
    bestAcc=acc
    }
    }

logit_fit <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 891)
## Train by all traing set
logit_fit_all <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 0)
# logit_fit_all <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, cross = 0)
logit_fit_all
## from W(weight), most infuential predictors: Pclass, gender,Embarked,wCabin
## base line: 0.8069585 (leave-one-out cv)


###############################################################

#### Use fewer predictor

pred_tr <- dataTr[,.(Pclass,wCabin,female)]
pred_tr <- data.matrix(pred_tr)
pred_tr <- cbind(pred_tr, dummy(dataTr$Embarked))
target <- as.factor(dataTr[, Survived])


tryCosts=c(200,150,100,50,30,1)
bestCost=NA
bestAcc=0
for(co in tryCosts){
    acc=LiblineaR(data=pred_tr,target=target,type=0,cost=co,bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 100)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
    bestCost=co
    bestAcc=acc
    }
    }

logit_fit <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 891)
## Train by all traing set
logit_fit_all <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 0)
# logit_fit_all <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, cross = 0)
logit_fit_all
## from W(weight), most infuential predictors: Pclass, gender,Embarked,wCabin
## base line: 0.8125701 (leave-one-out cv)




###############################################################

# Generate predictive result

pred_test <- dataTe[,.(Pclass,wCabin,female)]
pred_test <- data.matrix(pred_test)
pred_test <- cbind(pred_test, dummy(dataTe$Embarked))


result_test <- predict(logit_fit_all,pred_test,proba=T,decisionValues=TRUE)

# Output table
output_tb <- cbind(dataTe[,PassengerId], as.numeric(result_test$predictions)-1)
colnames(output_tb) <- c("PassengerId", "Survived") 
# 0.77990 accuracy on leader board

write.csv(output_tb, file="logit_pred3.csv", row.names = F)







##########################################################################
##########################################################################
##########################################################################



#### Random Forest

library(randomForest)

#trainset <- dataTr[,.(Survived,Pclass,Fare,wCabin,SibSp,Parch,relative,ticket_dup,tckPal,female,Age)]
trainset <- dataTr[,.(Survived,Pclass,Fare,wCabin,relative,tckPal,female,Age)]
trainset$Survived<-as.factor(trainset$Survived)

### run model
rf1<-randomForest(Survived~.,data=trainset,ntree=2000, mtry=3)
print(rf1)
# OOB estimate of  error rate: 16.61%

# importance of predictor
varImpPlot(rf1, main='') ## Importance: gender, age, pclass, fare; tckpal, relative, wcabin
impmx <- importance(rf1)


### predict using test set

# testset <- dataTe[,.(Pclass,Fare,wCabin,SibSp,Parch,relative,ticket_dup,tckPal,female,Age)]
testset <- dataTe[,.(Pclass,Fare,wCabin,relative,tckPal,female,Age)]

rf.pred1 <- predict( rf1, testset)   

# Output table
output_tb <- cbind(dataTe[,PassengerId], as.numeric(rf.pred1)-1)
colnames(output_tb) <- c("PassengerId", "Survived") 
# 0.76555 accuracy, worse than logistic and gender based model


write.csv(output_tb, file="rf1_pred4.csv", row.names = F)


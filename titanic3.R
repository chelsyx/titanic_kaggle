# Read in data
library(data.table)
data_train <- fread("data/train.csv",na.strings="")
data_test <- fread("data/test.csv",na.strings="")

data_all <- rbindlist(list(data_train,data_test),use.names=TRUE,fill=TRUE,idcol = "Source")


############## EXPLORE ########################
summary(data_all)

barplot(prop.table(table(data_train$Survived, data_train$Pclass)))

library(vcd)
mosaicplot(Survived~Pclass,data=data_train)

library(Hmisc)
bystats(data_train$Age,data_train$Embarked,fun=function(x)c(Mean=mean(x),Median=median(x)))


############## Preprocess ########################

## For imputation, stack predictors in training and test together

allPred <- rbindlist(list(data_train,data_test),use.names=TRUE,fill=TRUE,idcol = "Source")
ol.mask <- allPred$Fare==0
allPred$Fare[ol.mask]<-NA

# Impute Fare NAs by mean per pclass
fna.mask <- is.na(allPred$Fare)
aggregate(allPred$Fare[!fna.mask], by=list(allPred$Pclass[!fna.mask]), median)
allPred$Fare[fna.mask&allPred$Pclass==1]<-61.3792
allPred$Fare[fna.mask&allPred$Pclass==2]<-15.0500
allPred$Fare[fna.mask&allPred$Pclass==3]<-8.0500

# Impute Age NAs by median
ana.mask <- is.na(allPred$Age)
allPred$Age[ana.mask]<-median(allPred$Age[!ana.mask])

# Create New features
allPred$relative <- allPred$SibSp+allPred$Parch

allPred$ticket_dup <- as.numeric(duplicated(allPred$Ticket, fromLast=T)|duplicated(allPred$Ticket))
allPred$tckPal <- rep(1, nrow(allPred))
for (tckNum in unique(allPred$Ticket[allPred$ticket_dup==1])){
	xxpal.mask <- allPred$Ticket == tckNum
	allPred$tckPal[xxpal.mask] <- sum(xxpal.mask)
}
# Divide group fare by tckPal
allPred$Fare_ind <- allPred$Fare/allPred$tckPal

allPred$wCabin <- as.numeric(!is.na(allPred$Cabin))
allPred$cabinL <- substr(allPred$Cabin,0,1)
allPred$cabinL[is.na(allPred$cabinL)]<-"Unknown"
allPred$cabinL<-as.factor(allPred$cabinL)

# Transform
allPred$female <- as.numeric(allPred$Sex == "female")

allPred$Embarked[is.na(allPred$Embarked)] <- "S" ##impute by mode
allPred$Embarked <- as.factor(allPred$Embarked)

# Recover traing and test set
dataTr <- allPred[Source==1,]
dataTe <- allPred[Source==2,]
dataTe <- dataTe[,Survived:=NULL]

##########################################################################

##### Logistc Regression with regularizer

library(dummies)
library(LiblineaR)

pred_tr <- dataTr[,.(Pclass,Fare_ind,wCabin,SibSp,Parch,relative,ticket_dup,tckPal,female,Age)]
pred_tr <- data.matrix(pred_tr)
pred_tr <- cbind(pred_tr, dummy(dataTr$Embarked),dummy(dataTr$cabinL))
pred_tr_s <- scale(pred_tr)
target <- as.factor(dataTr[, Survived])


tryCosts=c(200,150,100,50,30,1)
bestCost=NA
bestAcc=0
for(co in tryCosts){
    acc=LiblineaR(data=pred_tr_s,target=target,type=0,cost=co,bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 100)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
    bestCost=co
    bestAcc=acc
    }
    }

logit_fit <- LiblineaR(pred_tr_s, target, type = 0, cost = 1, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 891)
## Train by all traing set
logit_fit_all <- LiblineaR(pred_tr_s, target, type = 0, cost = 1, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 0)
# logit_fit_all <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, cross = 0)
logit_fit_all
## from W(weight), most infuential predictors: Pclass, gender,Age;Sibsp,relative,pred_trE
## base line: 0.7676768 (leave-one-out cv)


###############################################################

#### Use fewer predictor

pred_tr <- dataTr[,.(Pclass,female,Age,SibSp,relative)]
pred_tr <- data.matrix(pred_tr)
pred_tr <- cbind(pred_tr, as.numeric(dataTr$cabinL=="E"))
colnames(pred_tr)[6]<-"cabinE"
pred_tr_s <- scale(pred_tr)
target <- as.factor(dataTr[, Survived])


tryCosts=c(200,150,100,50,30,1)
bestCost=NA
bestAcc=0
for(co in tryCosts){
    acc=LiblineaR(data=pred_tr_s,target=target,type=0,cost=co,bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 100)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
    bestCost=co
    bestAcc=acc
    }
    }

logit_fit <- LiblineaR(pred_tr_s, target, type = 0, cost = 1, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 891)
## Train by all traing set
logit_fit_all <- LiblineaR(pred_tr_s, target, type = 0, cost = 1, bias = F, wi = c("0"=0.6161616, "1"=0.3838384), cross = 0)
# logit_fit_all <- LiblineaR(pred_tr, target, type = 0, cost = 50, bias = F, cross = 0)
logit_fit_all
## base line: 0.7811448 (leave-one-out cv)




###############################################################

# Generate predictive result

pred_test <- dataTe[,.(Pclass,female,Age,SibSp,relative)]
pred_test <- data.matrix(pred_test)
pred_test <- cbind(pred_test, as.numeric(dataTe$cabinL=="E"))
colnames(pred_test)[6]<-"cabinE"
pred_test_s<-scale(pred_test,attr(pred_tr_s,"scaled:center"),attr(pred_tr_s,"scaled:scale"))

result_test <- predict(logit_fit_all,pred_test_s,proba=T,decisionValues=TRUE)

# Output table
output_tb <- cbind(dataTe[,PassengerId], as.numeric(result_test$predictions)-1)
colnames(output_tb) <- c("PassengerId", "Survived") 
# 0.73684 accuracy on leader board

write.csv(output_tb, file="logit_pred4.csv", row.names = F)







##########################################################################
##########################################################################
##########################################################################



#### Random Forest

library(party)


### run model
rf2 <- cforest(as.factor(Survived) ~ Pclass+Fare_ind+wCabin+SibSp+Parch+relative+ticket_dup+tckPal+female+Age+Embarked+cabinL,
               data = dataTr, controls=cforest_unbiased(ntree=2000, mtry=3))

print(rf2)

### predict using test set

testset <- dataTe[,.(Pclass,Fare_ind,wCabin,SibSp,Parch,relative,ticket_dup,tckPal,female,Age,Embarked,cabinL)]
# testset <- dataTe[,.(Pclass,Fare,wCabin,relative,tckPal,female,Age)]

Prediction <- predict(rf2, testset, OOB=TRUE, type = "response")


# Output table
output_tb <- cbind(dataTe[,PassengerId], as.numeric(Prediction)-1)
colnames(output_tb) <- c("PassengerId", "Survived") 
# 0.75598 accuracy, worse than logistic and gender based model


write.csv(output_tb, file="rf1_pred5.csv", row.names = F)


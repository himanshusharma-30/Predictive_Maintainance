### SETTING THE DATASETS ###

setwd("/home/Desktop/Link to Final Report/predictive maintenance")
train <- read.csv("predictive_maintenance_train.csv", header = TRUE)
valid <- read.csv("predictive_maintenance.csv", header = TRUE)

str(train)
##
## 'data.frame':	5999 obs. of  10 variables:
## $ X                      : int  3530 4502 1145 2016 443 4643 4410 4122 4571 4557 ...
## $ UDI                    : int  3530 4502 1145 2016 443 4643 4410 4122 4571 4557 ...
## $ Product.ID             : chr  "L50709" "L51681" "L48324" "L49195" ...
## $ Type                   : chr  "L" "L" "L" "L" ...
## $ Air.temperature..K.    : num  302 302 297 298 297 ...
## $ Process.temperature..K.: num  311 310 308 308 308 ...
## $ Rotational.speed..rpm. : int  1567 1307 1296 1301 1399 1238 1358 1364 1312 1349 ...
## $ Torque..Nm.            : num  39 54 69.1 66.3 61.5 54.6 54.6 47.8 52.2 51.2 ...
## $ Tool.wear..min.        : int  214 86 153 42 61 226 61 213 40 6 ...
## $ Target                 : int  1 1 1 1 1 1 1 1 1 1 ...

# As we can see the variable "Type" appear as character variable, but we need it as Factor variable. 
# Therefore, with the following command we will transform it in ordered factor. In addition, the variable 'Target' appears as
# a integer, so a transformation in factor it necessary.

train$Type <- factor(train$Type, ordered = TRUE, levels = c("L","M","H"))
train$Target <- as.factor(train$Target)

str(train)
## 'data.frame':	5999 obs. of  10 variables:
## $ X                      : int  3530 4502 1145 2016 443 4643 4410 4122 4571 4557 ...
## $ UDI                    : int  3530 4502 1145 2016 443 4643 4410 4122 4571 4557 ...
## $ Product.ID             : chr  "L50709" "L51681" "L48324" "L49195" ...
## $ Type                   : Ord.factor w/ 3 levels "L"<"M"<"H": 1 1 1 1 1 1 2 2 1 2 ...
## $ Air.temperature..K.    : num  302 302 297 298 297 ...
## $ Process.temperature..K.: num  311 310 308 308 308 ...
## $ Rotational.speed..rpm. : int  1567 1307 1296 1301 1399 1238 1358 1364 1312 1349 ...
## $ Torque..Nm.            : num  39 54 69.1 66.3 61.5 54.6 54.6 47.8 52.2 51.2 ...
## $ Tool.wear..min.        : int  214 86 153 42 61 226 61 213 40 6 ...
## $ Target                 : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...

# Now, in order to perform at best the various classification approaches, we will remove the identifier variables: X,
# UDI, Product.ID.

train$X <- NULL
train$UDI <- NULL
train$Product.ID <- NULL

str(train)
## 'data.frame':	5999 obs. of  7 variables:
## $ Type                   : Ord.factor w/ 3 levels "L"<"M"<"H": 1 1 1 1 1 1 2 2 1 2 ...
## $ Air.temperature..K.    : num  302 302 297 298 297 ...
## $ Process.temperature..K.: num  311 310 308 308 308 ...
## $ Rotational.speed..rpm. : int  1567 1307 1296 1301 1399 1238 1358 1364 1312 1349 ...
## $ Torque..Nm.            : num  39 54 69.1 66.3 61.5 54.6 54.6 47.8 52.2 51.2 ...
## $ Tool.wear..min.        : int  214 86 153 42 61 226 61 213 40 6 ...
## $ Target                 : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...

# In order to perform our analysis approaches, we will generate two different datasets from the train one: one excluding
# the Factor variable 'Type' and another one including it. Then we will perform the analysis approaches using both them 
# in order to evaluate hoe the accuracy is influenced by this variable.

# We apply the same changes to validation set and test set.

valid$Type <- factor(valid$Type, ordered = TRUE, levels = c("L","M","H"))
valid$Target <- as.factor(valid$Target)
valid$X <- NULL
valid$UDI <- NULL
valid$Product.ID <- NULL

str(valid)
## 'data.frame':	2002 obs. of  7 variables:
## $ Type                   : Ord.factor w/ 3 levels "L"<"M"<"H": 1 1 1 3 1 1 1 2 2 1 ...
## $ Air.temperature..K.    : num  298 297 296 297 299 ...
## $ Process.temperature..K.: num  308 308 306 308 310 ...
## $ Rotational.speed..rpm. : int  1348 1289 2270 1549 1371 1365 2886 1867 1542 1329 ...
## $ Torque..Nm.            : num  58.8 62 14.6 35.8 53.8 52.9 3.8 23.4 37.5 53.6 ...
## $ Tool.wear..min.        : int  202 199 149 206 228 218 57 225 203 207 ...
## $ Target                 : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...



### UNIVARIATE ANALYSIS ###

library("e1071")
library(GGally)
## Registered S3 method overwritten by 'GGally':
## method from   
## +.gg   ggplot2

# Check if there are some missing values
sum(is.na(train))
## [1] 0

# Air.temperature..K.

hist(train$Air.temperature..K., freq=FALSE, xlab= "Air temperature", col = "lightblue", main = "Histogram of Air.temperature..K.")
summary(train$Air.temperature..K.)
## Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## 295.4   298.3   300.1   300.0   301.5   304.5 

skewness(train$Air.temperature..K.)
## [1] 0.1219379

kurtosis(train$Air.temperature..K.)
## [1] -0.8442163


# Process.temperature..K.

hist(train$Process.temperature..K., freq=FALSE, xlab= "Process temperature K", col = "#FFC0CB", main = "Histogram of Process.temperature..K.")

summary(train$Process.temperature..K.)
##  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## 305.8   308.8   310.1   310.0   311.1   313.8 

skewness(train$Process.temperature..K.)
## [1] 0.02548641

kurtosis(train$Process.temperature..K.)
## [1] -0.5098995


# Rotational.speed..rpm.

hist(train$Rotational.speed..rpm., freq=FALSE, xlab= "Rotational speed rpm", col = "red", main = "Histogram of Rotational.speed..rpm.")

summary(train$Rotational.speed..rpm.)
## Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## 1181    1423    1504    1539    1611    2874 

skewness(train$Rotational.speed..rpm.)
## [1] 1.993557

kurtosis(train$Rotational.speed..rpm.)
## [1] 7.405396


# Torque..Nm.

hist(train$Torque..Nm., freq=FALSE, xlab= "Torque Nm ", col = "orange", main = "Histogram of Torque..Nm.")

summary(train$Torque..Nm.)
##  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## 4.20   33.20   40.10   40.01   46.80   76.20 

skewness(train$Torque..Nm.)
## [1] -0.003551766

kurtosis(train$Torque..Nm.)
## [1] -0.03281282


# Tool.wear..min.

hist(train$Tool.wear..min., freq=FALSE, xlab= "Tool wear min", col = "#FFD700", main = "Tool.wear..min.")

summary(train$Tool.wear..min.)
## Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## 0.0    53.0   109.0   108.2   162.0   253.0 

skewness(train$Tool.wear..min.)
## [1] 0.01914566

kurtosis(train$Tool.wear..min.)
## [1] -1.152117


# Type

table(train$Type)
##   L    M    H 
## 3625 1777  597 

round(table(train$Type)/length(train$Type)*100, digits=2)
##     L     M     H 
## 60.43 29.62  9.95 

barplot(table(train$Type)/length(train$Type)*100, col = c("green","blue", "red"), main = "percentages of Type")


# Target

table(train$Target)
##    0    1 
## 5796  203 

round(table(train$Target)/length(train$Target)*100, digits=2)
##     0     1 
## 96.62  3.38 

barplot(table(train$Target)/length(train$Target)*100, col = c("green","blue"), main = "percentages of Target")


### MULTIVARIATE ANALYSIS ###

train_quant_var <- train[, - 1]
Target<- as.factor(train$Target)
ggpairs(train_quant_var, aes(colour=Target))



### LOGISTIC REGRESSION ###

library(tidyverse)
library(MASS)
library(ROCR)
library(pROC)

# Logistic regression number 0 ALL Predictors

modelLr = glm(Target~., data=train, family="binomial")
summary(modelLr)
##
## Call:
## glm(formula = Target ~ ., family = "binomial", data = train)
## 
## Deviance Residuals: 
##   Min       1Q   Median       3Q      Max  
## -1.6598  -0.1850  -0.1007  -0.0543   3.6329  
## 
## Coefficients:
##                           Estimate Std. Error z value Pr(>|z|)    
## (Intercept)             -3.052e+01  1.905e+01  -1.602   0.1092    
## Type.L                  -6.575e-01  2.634e-01  -2.496   0.0125 *  
## Type.Q                  -1.279e-01  2.019e-01  -0.634   0.5264    
## Air.temperature..K.      7.587e-01  9.576e-02   7.923 2.32e-15 ***
## Process.temperature..K. -7.549e-01  1.279e-01  -5.901 3.61e-09 ***
## Rotational.speed..rpm.   1.209e-02  6.831e-04  17.695  < 2e-16 ***
## Torque..Nm.              2.906e-01  1.497e-02  19.409  < 2e-16 ***
## Tool.wear..min.          1.357e-02  1.479e-03   9.176  < 2e-16 ***
##   ---
## Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
## Null deviance: 1773.8  on 5998  degrees of freedom
## Residual deviance: 1127.6  on 5991  degrees of freedom
## AIC: 1143.6
## 
## Number of Fisher Scoring iterations: 8


# Logistic regression number 1 STEPAIC
#STEP AIC
modelLr1 <- glm(Target ~., data = train, family = binomial) %>%
  stepAIC(trace = TRUE)
## Start:  AIC=1143.65
## Target ~ Type + Air.temperature..K. + Process.temperature..K. + 
##    Rotational.speed..rpm. + Torque..Nm. + Tool.wear..min.
## 
##                           Df Deviance    AIC
## <none>                         1127.7 1143.7
## - Type                     2   1136.5 1148.5
## - Process.temperature..K.  1   1165.5 1179.5
## - Air.temperature..K.      1   1198.4 1212.4
## - Tool.wear..min.          1   1225.4 1239.4
## - Rotational.speed..rpm.   1   1406.9 1420.9
## - Torque..Nm.              1   1628.3 1642.3


# Summarize the final selected model
summary(modelLr1)
## 
## Call:
## glm(formula = Target ~ Type + Air.temperature..K. + Process.temperature..K. + 
## Rotational.speed..rpm. + Torque..Nm. + Tool.wear..min., family = binomial, 
## data = train)
## 
## Deviance Residuals: 
##   Min       1Q   Median       3Q      Max  
## -1.6598  -0.1850  -0.1007  -0.0543   3.6329  
## 
## Coefficients:
##                           Estimate Std. Error z value Pr(>|z|)    
## (Intercept)             -3.052e+01  1.905e+01  -1.602   0.1092    
## Type.L                  -6.575e-01  2.634e-01  -2.496   0.0125 *  
## Type.Q                  -1.279e-01  2.019e-01  -0.634   0.5264    
## Air.temperature..K.      7.587e-01  9.576e-02   7.923 2.32e-15 ***
## Process.temperature..K. -7.549e-01  1.279e-01  -5.901 3.61e-09 ***
## Rotational.speed..rpm.   1.209e-02  6.831e-04  17.695  < 2e-16 ***
## Torque..Nm.              2.906e-01  1.497e-02  19.409  < 2e-16 ***
## Tool.wear..min.          1.357e-02  1.479e-03   9.176  < 2e-16 ***
## ---
## Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
## Null deviance: 1773.8  on 5998  degrees of freedom
## Residual deviance: 1127.6  on 5991  degrees of freedom
## AIC: 1143.6
## 
## Number of Fisher Scoring iterations: 8
## 

# Testing prediction train set with threshold 0.5
predTrainLr = predict(modelLr1, type="response")

addmargins(table(predTrainLr>0.5,train$Target))
##          0    1  Sum
## FALSE 5776  158 5934
## TRUE    20   45   65
## Sum   5796  203 5999

predictedtrain <-as.numeric(predTrainLr > 0.5)
error_Lr_0.5 <- mean(predictedtrain != train$Target)
error_Lr_0.5
## [1] 0.02967161

accuracy_Lr_0.5 <- 1 - error_Lr_0.5
accuracy_Lr_0.5
## [1] 0.9703284

# ROC curve
ROCPred <- ROCR::prediction(predTrainLr, train$Target)
ROCPerf <- performance(ROCPred, "tpr", "fpr")
plot(ROCPerf,colorize=TRUE,lwd=2)
plot(ROCPerf,colorize=TRUE,lwd=2, print.cutoffs.at=c(0.05,0.2,0.5,0.8))
abline(a=0,b=1, lty=2)

# Best threshold value
my_roc <- roc(train$Target, predTrainLr)
## Setting levels: control = 0, case = 1
## Setting direction: controls < cases

coords(my_roc, "best", ret = "threshold")
##   threshold
## 1  0.050318

# AUC value
ROCauc <-performance(ROCPred, measure ="auc")
ROCauc@y.values[[1]]
## [1] 0.9059322

# testing prediction train set with threshold 0.1
addmargins(table(predTrainLr>0.1,train$Target))
##          0    1  Sum
## FALSE 5450   73 5523
## TRUE   346  130  476
## Sum   5796  203 5999

predictedtrain1 <-as.numeric(predTrainLr > 0.1)
error_Lr_0.1 <-  mean(predictedtrain1 != train$Target)
error_Lr_0.1
## [1] 0.06984497

accuracy_Lr_0.1 <- 1 - mean(predictedtrain1 != train$Target)
accuracy_Lr_0.1
## [1] 0.930155

# Testing prediction w/ thr 0.05 validation set 
predTestLr = predict(modelLr1, newdata=valid, type="response")
addmargins(table(predTestLr>0.05,valid$Target))
##        
##          0    1  Sum
## FALSE 1689   21 1710
## TRUE   244   48  292
## Sum   1933   69 2002

predictedtest <-as.numeric(predTestLr > 0.05)
error_Lr.v0.05 <- mean(predictedtest != valid$Target)
error_Lr.v0.05
## [1] 0.1323676
accuracy_Lr.v0.05 <- 1 - error_Lr.v
accuracy_Lr.v0.05
## [1] 0.8676324

# Testing prediction w/ thr 0.1 validation set 
predTestLr = predict(modelLr1, newdata=valid, type="response")
addmargins(table(predTestLr>0.1,valid$Target))
##       
## 0    1  Sum
## FALSE 1835   28 1863
## TRUE    98   41  139
## Sum   1933   69 2002

predictedtest0.1 <-as.numeric(predTestLr > 0.1)
error_Lr.v0.1 <- mean(predictedtest0.1 != valid$Target)
error_Lr.v0.1
## [1] 0.06293706
accuracy_Lr.v0.1 <- 1 - error_Lr.v0.1
accuracy_Lr.v0.1
## [1] 0.9370629



### RANDOM FOREST ###

library(randomForest)
##randomForest 4.7-1.1
##Type rfNews() to see new features/changes/bug fixes.

# The random forest is a ensemble method that use $sqrt(p)$ predictors, therefore in our case we will set mtry = 3.

set.seed(13)
df <- rbind(train, valid)

rf <- randomForest(Target ~ ., data = df,
                   mtry = 3, importance = TRUE)
rf
##
## Call:
## randomForest(formula = Target ~ ., data = df, mtry = 3, importance = TRUE) 
## Type of random forest: classification
## Number of trees: 500
## No. of variables tried at each split: 3
## 
##        OOB estimate of  error rate: 1.31%
## Confusion matrix:
##      0   1 class.error
## 0 7706  23 0.002975805
## 1   82 190 0.301470588

accuracy.rf <- (rf$confusion[1,1] + rf$confusion[2,2]) / (rf$confusion[1,1] + rf$confusion[1,2]
                                                          + rf$confusion[2,1] + rf$confusion[2,2])
accuracy.rf
# [1] 0.9868766

# Now we can visualize the importance of each variable in the built random forest.

importance(rf)
##                                 0         1 MeanDecreaseAccuracy MeanDecreaseGini
## Type                     8.717722 18.078235             17.12808         13.42436
## Air.temperature..K.     48.112550 66.950043             60.45186         80.76690
## Process.temperature..K. 41.597453  7.096397             44.67121         73.10198
## Rotational.speed..rpm.  45.103999 50.025417             57.03827        100.58031
## Torque..Nm.             53.084299 88.550556             69.59543        170.90059
## Tool.wear..min.         46.422980 67.924776             70.05821         85.80741

varImpPlot(rf, main = "Importance of each variable")


### WITH STRATIFIED SAMPLING ###
k <- length(df$Target[df$Target == 1])
stratified.rf = randomForest(Target ~ ., data = df, strata = df$Target, sampsize = c(10*k,k) ,
                             mtry = 3, importance = TRUE)
stratified.rf
## 
## Call:
##   randomForest(formula = Target ~ ., data = df, strata = df$Target, sampsize = c(10 * k, k), mtry = 3, importance = TRUE) 
## Type of random forest: classification
## Number of trees: 500
## No. of variables tried at each split: 3
## 
## OOB estimate of  error rate: 1.44%
## Confusion matrix:
##      0   1 class.error
## 0 7677  52 0.006727908
## 1   63 209 0.231617647

accuracy.stratified.rf <- (stratified.rf$confusion[1,1] + stratified.rf$confusion[2,2]) / 
  (stratified.rf$confusion[1,1] + stratified.rf$confusion[1,2] + stratified.rf$confusion[2,1]
   + stratified.rf$confusion[2,2])
accuracy.stratified.rf
## [1] 0.9856268

importance(stratified.rf)
##                                 0         1 MeanDecreaseAccuracy MeanDecreaseGini
## Type                     7.755253 16.905939             13.74844         9.095684
## Air.temperature..K.     32.110556 66.266869             36.79287        72.592087
## Process.temperature..K. 26.301736  8.838071             27.52204        50.980035
## Rotational.speed..rpm.  38.417083 49.496883             45.11411       111.817435
## Torque..Nm.             40.594234 77.632196             49.74724       163.552538
## Tool.wear..min.         46.462691 72.283030             66.29542        86.490675

varImpPlot(stratified.rf, main = "Importance of each variable")




### NEURAL NETWORK ###

library(keras)
tensorflow::set_random_seed(13)
#Loaded Tensorflow version 2.9.1


scaled.train <- scale(model.matrix(Target ~. - 1, data = train))
true.train.targets <- to_categorical(train$Target)
scaled.valid <- scale(model.matrix(Target ~. - 1, data = valid))
true.valid.targets <- to_categorical(valid$Target)

modelnn <-  keras_model_sequential()
modelnn %>%
  layer_dense(units = 10, activation = 'relu', input_shape = ncol(scaled.train)) %>%
  layer_dense(units = 7, activation = 'relu')  %>%
  layer_dense(units = 2, activation = "sigmoid")

summary(modelnn)  
##Model: "sequential"
##______________________________________________________________________________________________________________________
##  Layer (type)                                        Output Shape                                   Param #           
##======================================================================================================================
##  dense_2 (Dense)                                     (None, 10)                                     90                
##  dense_1 (Dense)                                     (None, 7)                                      77                
##  dense (Dense)                                       (None, 2)                                      16                
##======================================================================================================================
## Total params: 183
## Trainable params: 183
## Non-trainable params: 0
##______________________________________________________________________________________________________________________

modelnn %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  metrics = metric_binary_accuracy
)

system.time(
  history <- modelnn %>% fit(
    scaled.train,
    true.train.targets,
    epochs = 50,
    batch_size = 32,
    validation_data = list(scaled.valid, true.valid.targets)
  )
)


plot(history)

modelnn %>% evaluate(scaled.train, true.train.targets)
#188/188 [==============================] - 0s 702us/step - loss: 0.0717 - binary_accuracy: 0.9762
#           loss binary_accuracy 
#     0.07166519      0.97616267

modelnn %>% evaluate(scaled.valid, true.valid.targets)
#63/63 [==============================] - 0s 2ms/step - loss: 0.0845 - binary_accuracy: 0.9740
#          loss binary_accuracy 
#    0.0845117       0.9740260 


# Confusion matrix on validation
pred<-as.matrix(modelnn %>% predict(scaled.valid) %>% `>`(0.5) %>% k_cast("int32"))
## 63/63 [==============================] - 0s 643us/step

table(pred[,2], true.valid.targets[,2]) 
## 
##      0    1
## 0 1929   51
## 1    4   18

### APPLAYING THE MODEL TO THE TEST DATASET ###

test <- read.csv("predictive_maintenance_test.csv", header = TRUE)

trained.test <- test
trained.test$X <- NULL
trained.test$UDI <- NULL
trained.test$Product.ID <- NULL
trained.test$id_number <- NULL
trained.test$Type <- factor(test$Type, ordered = TRUE, levels = c("L","M","H"))

str(trained.test)
## 'data.frame':	1999 obs. of  6 variables:
## $ Type                   : Ord.factor w/ 3 levels "L"<"M"<"H": 1 1 2 1 2 1 2 1 1 1 ...
## $ Air.temperature..K.    : num  302 302 301 299 302 ...
## $ Process.temperature..K.: num  310 311 310 310 310 ...
## $ Rotational.speed..rpm. : int  1351 1270 1996 1377 1284 1372 1671 1326 1312 1316 ...
## $ Torque..Nm.            : num  45.1 65.3 19.8 62.5 68.2 60.1 30.5 58.5 65.3 61.2 ...
## $ Tool.wear..min.        : int  168 182 203 92 111 212 234 55 192 200 ...

pred.target = predict(stratified.rf, newdata=trained.test, type="response")
test$Target <- pred.target
head(test)
## 
## X  UDI Product.ID Type Air.temperature..K. Process.temperature..K.
## 1 4537 4537     L51716    L               302.4                   310.2
## 2 3685 3685     L50864    L               302.0                   311.2
## 3 2942 2942     M17801    M               300.7                   309.6
## 4 9614 9614     L56793    L               299.0                   310.2
## 5 4343 4343     M19202    M               301.7                   309.8
## 6 7510 7510     L54689    L               300.6                   311.9
## Rotational.speed..rpm. Torque..Nm. Tool.wear..min. id_number Target
## 1                   1351        45.1             168         1      1
## 2                   1270        65.3             182         2      1
## 3                   1996        19.8             203         3      0
## 4                   1377        62.5              92         4      0
## 5                   1284        68.2             111         5      1
## 6                   1372        60.1             212         6      1

write.csv(test, file = "final data frame.csv")

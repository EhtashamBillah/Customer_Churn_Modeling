


# Importing the dataset
dataset <- read.csv('Churn_Modelling.csv')
# keeping the necessary columns
dataset <- dataset[,4:14]
str(dataset)
# Creating the dummy variable for Geography and Gender variable
dataset$Geography <-factor(dataset$Geography,levels=c("France","Spain","Germany"),labels =c(1,2,3))
dataset$Gender <-factor(dataset$Gender,levels=c("Female","Male"),labels =c(1,2))
# XGBOOST want to treat the dummy variable as numeric value.so we need to transform it to numeric
dataset$Geography <-as.numeric(dataset$Geography)
dataset$Gender <-as.numeric(dataset$Gender)


# Splitting the dataset into the Training set and Test set
require(caTools)
set.seed(123)
split <- sample.split(dataset$Exited,SplitRatio = 0.80)
training_set <- subset(dataset,split==T)
test_set <- subset(dataset,split==F)

# creating XGBOOST model from training set/ fitting XGBOOST to training set
#install.packages("xgboost")
require(xgboost)
model_xgboost <- xgboost(data = as.matrix(training_set[,-11]), label = training_set$Exited, nrounds = 10) #nrounds=10 meansd we want 10 iterations

# we want to EVALUATE xgboost model using k-fold cross validation
require(caret)
folds <- createFolds(training_set$Exited, k=10) # 10 different test folds composing training set
folds
cv <- lapply(folds,function(x){ # x is each observation of all 10 test folds
  training_fold <- training_set[-x,] # whole training set except the 10 test fold
  test_fold <- training_set[x,]
  model_xgboost <- xgboost(data=as.matrix(training_set[,-11]),label = training_set$Exited,nrounds=10) #nrounds=10 meansd we want 10 iterations
  y_hat <- predict(model_xgboost, newdata = as.matrix(test_fold[,-11])) # test fold contain dep var, so omit it
  y_hat <- ifelse(y_hat>0.5,1,0)
  cm <-table(test_fold[,11],y_hat)
  accuracy <- (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1]) # accuracy of predicting y from 10 test folds
  return(accuracy)
})
mean(as.numeric(cv))


# pridicting y of test set
y_hat <- predict(model_xgboost, newdata = as.matrix(test_set[,-11]))
y_hat <- ifelse(y_hat>0.5,1,0)
y_hat

# confussin matrix
cm <- table(test_set[,11],y_hat)
cm
#accuracy
accuracy <- (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
accuracy

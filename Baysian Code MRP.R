#Install packages
install.packages("bnlearn")
install.packages("foreach")
install.packages("parallel")
install.packages("doParallel")
install.packages("ROCR")
install.packages("data.table")
install.packages("ggplot2")

#Load library 
library(bnlearn)
library(foreach)
library(parallel)
library(doParallel)
library(ROCR)
library(data.table)
library(ggplot2)

# Load Data
mydata <- read.csv("creditCard.csv")

# Sampling data
df <- mydata

#Feature Normalization
normalize <- function(x) {
  return( (x - min(x))/ (max(x)- min(x)))
}

x <- df$Class
df <- as.data.frame(lapply(df[,c(2:30)], normalize))
df$Class <- x

#Feature Discretization
list <- c(-0.1,0.25,0.50,0.75,1)


list2 <- lapply(list, as.character)
list2 <-list2[-1]

df$V1 <- cut(df$V1,br=list, labels =list2)
df$V2 <- cut(df$V2,br=list, labels =list2)
df$V3 <- cut(df$V3,br=list, labels =list2)
df$V4 <- cut(df$V4,br=list, labels =list2)
df$V5 <- cut(df$V5,br=list, labels =list2)
df$V6 <- cut(df$V6,br=list, labels =list2)
df$V7 <- cut(df$V7,br=list, labels =list2)
df$V8 <- cut(df$V8,br=list, labels =list2)
df$V9 <- cut(df$V9,br=list, labels =list2)
df$V10 <- cut(df$V10,br=list, labels =list2)
df$V11 <- cut(df$V11,br=list, labels =list2)
df$V12 <- cut(df$V12,br=list, labels =list2)
df$V13 <- cut(df$V13,br=list, labels =list2)
df$Amount <- cut(df$Amount,br=list, labels =list2)
df$Class <- as.factor(df$Class)

str(df)


#Remove unused Features 
df <- df[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,29,30)]



#Shuffle dataset
set.seed(145)
gp <- runif(nrow(df))  
df <- df[order(gp),]

#Split into training/testing sets
train_data <- df[1:250000,]
test_data <- df[250001:284807,]

table(train_data$Class)
table(test_data$Class)


# Learn the structure of a Bayesian network using a hill-climbing greedy search.
res <- hc(train_data)

# Plot the nodes of the DAG with their directed edges
plot(res)

#Changes Edges
res$arcs

res$arcs[1,2] <- "Class"
res$arcs[2,2] <- "Class"
res$arcs[3,2] <- "Class"
res$arcs[6,2] <- "Class"
res$arcs[8,2] <- "Class"
res$arcs[9,2] <- "Class"
res$arcs[10,2] <- "Class"
res$arcs[25,2] <- "Class"
res$arcs[4,1] <- "V8"
res$arcs[4,2] <- "Class"
res$arcs[5,1] <- "V4"
res$arcs[5,2] <- "Class"
res$arcs[7,1] <- "V13"
res$arcs[7,2] <- "Class"
res$arcs[14,1] <- "Amount"
res$arcs[14,2] <- "Class"

# Fit the parameters of a Bayesian network conditional on its structure
fittedbn <- bn.fit(res, data = train_data)

#set Parameters
numDraws <- 100000
set.seed(130)
list <- c()

#Run Bayesian Network on testing data
begin.time <- Sys.time()

for (i in c(1:34807)){
  
x <- cpquery(fittedbn, event = (Class == "1"), 
        evidence =(V1==test_data[i,1]&V2==test_data[i,2]&V3==test_data[i,3]&V4==test_data[i,4]&V5==test_data[i,5]
                   &V6==test_data[i,6]&V7==test_data[i,7]
                   &V8==test_data[i,8]&V9==test_data[i,9]
                   &V10==test_data[i,10]&V11==test_data[i,11]
                   &V12==test_data[i,12]&V13==test_data[i,13]
                   &Amount==test_data[i,14]), n = numDraws)

  list <- c(list,x)
}
timetaken(begin.time)

#Convert the probabilities
list2 <- list
list <- list2
list <- data.frame(list)
colnames(list) <-"prob"

list$prob[list$prob >= 0.35]<- 1
list$prob[list$prob != 1] <- 0

table(list)

#Validation Metrics confusion Matrix, Precison, Recall, ROC  
table(test_data$Class,list$prob)
mean(list$prob == test_data$Class)

#ROC
pred <- prediction(list,test_data$Class)
roc <- performance(pred,"tpr","fpr")
plot(roc, main = "ROC Curve")
abline(a=0, b=1, col = "red")

#AUC
auc <- performance(pred, "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6,.4,auc, title = "AUC")


#Parallel Implementation
detectCores()
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
clusterExport(cl,"test_data")
clusterExport(cl,"i")
registerDoParallel(cl)


#set Parameters
numDraws <- 100000
set.seed(130)
list <- c()

#Bayesian Network Parallel

#test_data <- data.frame(lapply(test_data, as.character))
begin.time <- Sys.time()
for (i in c(1:34807)){
  clusterExport(cl,"i")
  x <- cpquery(fittedbn, event = (Class == "1"), 
               evidence =(V1==test_data[i,1]&V2==test_data[i,2]&V3==test_data[i,3]&V4==test_data[i,4]&V5==test_data[i,5]
                          &V6==test_data[i,6]&V7==test_data[i,7]
                          &V8==test_data[i,8]&V9==test_data[i,9]
                          &V10==test_data[i,10]&V11==test_data[i,11]
                          &V12==test_data[i,12]&V13==test_data[i,13]
                          &Amount==test_data[i,14]), n = numDraws, cluster = cl)
  
  list <- c(list,x)
  
  
}
timetaken(begin.time)
stopCluster(cl)


#Convert the probabilities
list2 <- list
list <- list2
list <- data.frame(list)
colnames(list) <-"prob"

list$prob[list$prob >= 0.3]<- 1
list$prob[list$prob != 1] <- 0

table(list)

#Validation Metrics confusion Matrix, Precison, Recall  
table(test_data$Class,list$prob)
mean(list$prob == test_data$Class)

#Parallel / Non-Parallel comparison 
normal <- c(360+14,420+27,612,842,1027,1252,1740+28,2220+48,3660+30)
Parallel <- c(420+52,480+1,667,788,960+23,1060+53,1220+28,1540+32,2080+55)
Indexes <- c(300,1000,10000,15000,21000,30000,45000,60000,100000)

df_plot <- data.frame(
  Legend = factor(c("Non-Parallel","Non-Parallel","Non-Parallel","Non-Parallel","Non-Parallel","Non-Parallel","Non-Parallel","Non-Parallel","Non-Parallel","Parallel","Parallel","Parallel","Parallel","Parallel","Parallel","Parallel","Parallel","Parallel")),
  Indexes = factor(c(300,1000,10000,15000,21000,30000,45000,60000,100000,300,1000,10000,15000,21000,30000,45000,60000,100000),
  levels =c(300,1000,10000,15000,21000,30000,45000,60000,100000)),
  Time =c(normal,Parallel)
)

p1 <- ggplot(data= df_plot, aes(x=Indexes, y=Time, group=Legend, shape = Legend, colour= Legend)) +
        geom_line() + 
        geom_point() + 
        xlab("Iterations") +
        ylab("Run Time (Seconds)")
       
p1




#############################################################################################################################
#END - EXTRA Analysis
############################################################################################################################

#Data Exploration (V1)
par(mfrow=c(1,2))
x <- df$V1[df$Class==1]
plot(x)
abline(h=-7.5, col="blue")
abline(h=2.132386, col="blue")
table(df$Class)

df <- subset(df, (df$V1 >= -7.5 & df$V1 < 2.132386 & df$Class ==1) | df$Class == 0)


df$V1[df$V1 >= -7.5 & df$V1 <= 2.132386] <- 1
df$V1[df$V1 != 1] <- 0
table(df$V1)


#Data Exploration (V2)
x <- df$V2[df$Class==1]
plot(x)
abline(h=0, col="blue")
abline(h=6.4, col="blue")

df <- subset(df, (df$V2 >= 0 & df$V2 < 6.4 & df$Class ==1) | df$Class == 0)


df$V2[df$V2 >= 0 & df$V2 <= 6.4] <- 1
df$V2[df$V2 != 1] <- 0
table(df$V2)

#Data Exploration (V3)
x <- df$V3[df$Class==1]
plot(x)
abline(h=0, col="blue")
abline(h=-8, col="blue")

df <- subset(df, (df$V3 <= 0 & df$V3 >= -8 & df$Class ==1) | df$Class == 0)


df$V3[df$V3 <= 0 & df$V3 >= -8] <- 1
df$V3[df$V3 != 1] <- 0
table(df$V3)

#Data Exploration (V4)
x <- df$V4[df$Class==1]
plot(x)

abline(h=-.67, col="blue")
abline(h=7.2, col="blue")


df <- subset(df, (df$V4 >= -0.67 & df$V4 <= 7.2 & df$Class ==1) | df$Class == 0)

df$V4[df$V4 >= -0.67 & df$V4 <= 7.2] <- 1
df$V4[df$V4 != 1] <- 0
table(df$V4)

#Data Exploration (V5)
x <- df$V5[df$Class==1]
plot(x)

abline(h=2, col="blue")
abline(h=-5, col="blue")


df <- subset(df, (df$V5 >= -4 & df$V5 <= 2 & df$Class ==1) | df$Class == 0)

df$V5[df$V5 >= -4 & df$V5 <= 2] <- 1
df$V5[df$V5 != 1] <- 0
table(df$V5)

#Data Exploration (V6)
x <- df$V6[df$Class==1]
plot(x)

abline(h=0, col="blue")
abline(h=-3, col="blue")


df <- subset(df, (df$V6 >= -3 & df$V6 <= 0 & df$Class ==1) | df$Class == 0)

df$V6[df$V6 >= -3 & df$V6 <= 0] <- 1
df$V6[df$V6 != 1] <- 0
table(df$V6)

#Data Exploration (V7)
x <- df$V7[df$Class==1]
plot(x)

abline(h=0, col="blue")
abline(h=-6, col="blue")


df <- subset(df, (df$V7 >= -6 & df$V7 <= 0 & df$Class ==1) | df$Class == 0)

df$V7[df$V7 >= -6 & df$V7 <= 0] <- 1
df$V7[df$V7 != 1] <- 0
table(df$V7)

#Data Exploration (V8)
x <- df$V8[df$Class==1]
plot(x)

abline(h=2, col="blue")
abline(h=-0.5, col="blue")


df <- subset(df, (df$V8 >= -0.5 & df$V8 <= 2 & df$Class ==1) | df$Class == 0)

df$V8[df$V8 >= -0.5 & df$V8 <= 2] <- 1
df$V8[df$V8 != 1] <- 0
table(df$V8)

#Data Exploration (V9)
x <- df$V9[df$Class==1]
plot(x)

abline(h=0, col="blue")
abline(h=-3, col="blue")


df <- subset(df, (df$V9 >= -3 & df$V9 <= 0 & df$Class ==1) | df$Class == 0)

df$V9[df$V9 >= -3 & df$V9 <= 2] <- 1
df$V9[df$V9 != 1] <- 0
table(df$V9)

#Data Exploration (V10)
x <- df$V10[df$Class==1]
plot(x)

abline(h=-2, col="blue")
abline(h=-6, col="blue")


df <- subset(df, (df$V10 >= -6 & df$V10 <= -2 & df$Class ==1) | df$Class == 0)

df$V10[df$V10 >= -6 & df$V10 <= 2] <- 1
df$V10[df$V10 != 1] <- 0
table(df$V10)

#Data Exploration (Amount)
x <- df$Amount[df$Class==1]
plot(x)

abline(h=0, col="blue")
abline(h=200, col="blue")


df <- subset(df, (df$Amount >= 0 & df$Amount <= 200 & df$Class ==1) | df$Class == 0)

df$Amount[df$Amount >= 0 & df$Amount <= 200] <- 1
df$Amount[df$Amount != 1] <- 0
table(df$Amount)

# Loop to convert data to binary 

for (i in c(2:29)){
  med <- median(df[,i])
  
  if (med >= 0 ){
    df[,i][df[,i] > med] <- 1
    df[,i][df[,i] <= med] <- 0 
  }
  else {
    df[,i][df[,i] > med] <- 1
    df[,i][df[,i] <= med] <- 0 
    
  }
}


#Na checker
count = 0
for (i in c(1:284807)){
  for (p in c(1:15)){
    if (is.na(df[i,p])){
      count = count + 1 
    }
    
  }
}
count

# Remove edges of the graph that does not seem to be relevant
res$arcs
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V3" & res$arcs[,'to'] == "Amount")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V10" & res$arcs[,'to'] == "V4")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V6" & res$arcs[,'to'] == "V10")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V8" & res$arcs[,'to'] == "V9")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V6" & res$arcs[,'to'] == "V8")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V3" & res$arcs[,'to'] == "V12")),]

res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V7" & res$arcs[,'to'] == "V11")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V2" & res$arcs[,'to'] == "V4")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V3" & res$arcs[,'to'] == "V6")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V12" & res$arcs[,'to'] == "V10")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V12" & res$arcs[,'to'] == "V13")),]
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V12" & res$arcs[,'to'] == "V4")),]

res$arcs
res$arcs <- res$arcs[-which((res$arcs[,'from'] == "V10" & res$arcs[,'to'] == "V4")),]

res$arcs[23,1] <- "V5"
res$arcs[23,2] <- "Amount"

plot(res)

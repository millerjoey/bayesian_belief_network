# =============================================
# Bayesian Belief Network to classify FARS Data
# =============================================

# Dataset from: http://www-fars.nhtsa.dot.gov//QueryTool/QuerySection/SelectYear.aspx

library(dplyr)
library(bnlearn)
library(gRain)


setwd("~/Desktop/BBN_afterinc/data/")

data <- read.table("interesting_features.txt", header = T, sep = "\t", na.strings = ".")
data <- data[,-55] # Empty column
data <- filter(data, pnumber==1) # Look at drivers only.
fars <- select(data, -Obs., -casenum, -pnumber) # Eliminate low-info variables. Keeping vnumber.
rm(data)

# ====Cleaning Data, dridistract=====
save.image("slide5")
load("slide5")
strings <- strsplit(as.character(fars$dridistract), split = " ")

l <- vector(mode="integer", length=length(strings))
for (i in 1:length(strings)) { # Get vector with lengths of strings. The 2s correspond to rows with more than 1 entry.
  l[i] <- length(strings[[i]])
}

for(j in 1:max(l)) { # Getting new columns 1,2,3
  col <- vector(mode="integer", length=length(strings))
  for (i in 1:length(strings)) {
    col[i] <- as.integer(strings[[i]][j])
  }
  assign(paste("dridistract", j, sep = ""), value = col)
}

fars$dridistract <- dridistract1
fars <- cbind.data.frame(fars, dridistract2, dridistract3)

# =====Cleaning Data, drivisobs==== (will just repeat)
strings <- strsplit(as.character(fars$drivisobs), split = " ")

l <- vector(mode="integer", length=length(strings))
for (i in 1:length(strings)) { # Get vector with lengths of strings. The 2s correspond to rows with more than 1 entry.
  l[i] <- length(strings[[i]])
}

for(j in 1:max(l)) { # Getting new columns 1,2,3
  col <- vector(mode="integer", length=length(strings))
  for (i in 1:length(strings)) {
    col[i] <- as.integer(strings[[i]][j])
  }
  assign(paste("drivisobs", j, sep = ""), value = col)
}

fars$drivisobs <- drivisobs1
fars <- cbind.data.frame(fars, drivisobs2, drivisobs3)

rm(strings, i, j, l, col)
rm(list=ls()[1:6])

# ====Dealing with Height===
save.image("line64")
load("line64")
convertHeight <- function(x) {
  # inches is at index 44
  # feet is at index 43
  if (x[43]==9 | x[44]==99 | any(is.na(x[43]))) {
    height <- NA
  }
  else {
    height <- 12*x[43]+x[44]
  }
  return(height)
}

height <- apply(fars, MARGIN = 1, FUN = convertHeight)
rm(convertHeight)
fars <- cbind.data.frame(fars, height)
fars <- select(fars, -inches, -feet)
rm(height)
# ====Replace NA with -1 (of categorical vars only; not height, spdlim, age)====
temp <- select(fars, -height, -spdlim, -age)
temp <- as.data.frame(apply(X = temp, MARGIN=2, FUN = function(x) replace(x,which(is.na(x)),-1)))
fars <- cbind.data.frame(temp, select(fars, height, spdlim, age))
rm(temp)

# ===Final Profiling before I Discretize===
num_categories <- vector(mode = "integer")
for (i in 1:dim(fars)[2]) {
  num_categories[i] <- length(unique(fars[,i]))
}
names(num_categories) <- names(fars)
# num_categories contains the Counts of each number of levels. 
# will use to index variables for discretize()

levels <- list()
for (i in 1:dim(fars)[2]) {
  levels[[i]] <- unique(fars[,i])
}
names(levels) <- names(fars)
# levels is a list with the possible levels of each feature. Can use later for categorization after discretize()

save.image("line105")
load("line105")
# =====Feature Selection=====
library(caret)

cat_data <- select(fars, -age, -height, -spdlim, -travspd)

x <- nzv(cat_data, names = T)
tables <- list()
for (i in x) {
  tables[[i]] <-table(cat_data[,i])
}

# Will ignore those variables. Low variability, can't train easily on it.
rm(cat_data)
x <- which(names(fars) %in% x)
fars <- fars[,-x]

# ====Load From Here=====
save.image("r_code_env")
load("r_code_env")

tables <- list()
for (i in names(fars)) {
  tables[[i]] <-table(fars[,i])
}

# ===Merging Categories===

metricVars <- c("age", "height", "spdlim", "travspd", "alcres", "fatcount")
metricVarsI <- which(names(fars) %in% metricVars)
cdata <- fars[,-metricVarsI]
mdata <- fars[,metricVarsI]

fdata <- as.data.frame(apply(X=cdata, MARGIN=2, FUN = as.factor))

merge_categories <- function(fdata, p) {
  merged_levs <- list()
  for (col in 1:dim(fdata)[2]) {
    dat <- fdata[,col]
    lowfreq <- names(which(prop.table(table(dat)) < p))
    levels(dat)[levels(dat) %in% lowfreq] <- "Other"
    merged_levs[[names(fdata)[col]]] <- lowfreq
    fdata[,col] <- dat
  }
  return(list(fdata, merged_levs))
}

out <- merge_categories(fdata, 0.01)
fdata <- out[[1]]
merged_levs <- out[[2]]

rm(metricVars, metricVarsI, out, x)
save.image("line157")
load("line157")

# ---Discretizing Metric Data---
library(bnlearn)

fars <- cbind.data.frame(mdata, cdata$vfatcount)

temp <- as.data.frame(apply(X = fars, MARGIN=2, FUN = function(x) replace(x,which(is.na(x)),max(x, na.rm = T))))
temp <- as.data.frame(apply(X = temp, MARGIN=2, FUN = function(x) x+runif(n = length(x), min = -0.001, max=0.001)))

# Hartemink method: cuts numeric data into 30 quantiles (ibreaks) and combines into 3 (breaks) in a way that reduces the mutual information
# between each variable and the other variables minimally. You will NOT get equally sized intervals in count or measure. Takes ~10s of seconds.
mdata <- discretize(data = temp, method = "hartemink", breaks = 5, ibreaks = 30, idisc="quantile")
mdata <- mdata[,-6]
rm(temp)

# ---Combine Data Frames---
fars <- cbind.data.frame(mdata, fdata)
save.image("line173")
setwd("Desktop/BBN_afterinc/data/")
load("line173")
# ===Structure Learning===

rm(cdata, fdata, mdata, metricVars, metricVarsI)
set.seed(0602)
rand.rows <- sample(size = round(0.2*dim(fars)[1]), x = (1:dim(fars)[1]), replace = F)
train <- fars[-rand.rows,]
test <- fars[rand.rows,]

save.image("line188")
setwd("Desktop/BBN_afterinc/data/")
load("line188"); library(bnlearn)

# --Hill Climbing BIC---
bn_hc <- hc(x=train, score="bic")
graphviz.plot(bn_hc, shape="ellipse")
bnhc_fit <- bn.fit(bn_hc, train, method = "bayes")
#------------------

# ---Hill Climbing Bayesian---
bn_hc2 <- hc(x = train, score = "bde", iss=1)
graphviz.plot(bn_hc2, shape="ellipse")
bnhc2_fit <- bn.fit(bn_hc2, train, method = "bayes")
# ------------------

# Images:
graphviz.plot(bn_hc, shape="ellipse", highlight = list('nodes'=c(mb1, "vfatcount"),
                                                        'fill'=c(rep("black", times=length(mb1)), "red"),
                                                        'textCol'=c(rep("white", times=length(mb1)+1)),
                                                        'col'=c("black")))
dev.print(pdf, "bn_hc")

graphviz.plot(bn_hc2, shape="ellipse", highlight = list('nodes'=c(mb2, "vfatcount"),
                                                       'fill'=c(rep("black", times=length(mb2)), "red"),
                                                       'textCol'=c(rep("white", times=length(mb2)+1)),
                                                       'col'=c("black")))
dev.print(pdf, "bn_hc2")

# Predictions (markov blanket) and confusion matrix.

bn_gR1 <- as.grain(bnhc_fit)
bn_gR2 <- as.grain(bnhc2_fit)
mb1 <- mb(bn_hc, "vfatcount")
mb2 <- mb(bn_hc2, "vfatcount")

library(caret)
library(gRain)

library(caret)
library(bnlearn)
library(gRain)
bn1 <- as.grain(bnhc_fit)
bn2 <- as.grain(bnhc2_fit)

evaluate <- function(bn, predictors=F, target, data, type = "class") {
  if (all(predictors==F)) {predictors <- mb(as.bn.fit(bn), target)} # If you leave predictors false, use markov blanket. All() so it doens't error if length(predictors)>1.
  t0 <- proc.time()
  out <- list()
  preds <- predict.grain(object = bn, response = target, predictors = predictors, newdata = data, type = type)
  out[["preds"]] <- preds$pred[[1]]
  if (type=="class") {
    tf <- unlist(preds[[1]])==as.character(data[,target])
    out[["prop"]] <- sum(tf)/length(tf)
  }
  out[["observed"]] <- as.character(data[,target])
  t1 <- proc.time()
  out[["time"]] <- t1-t0
  
  confusion <- function(evals) {
    l1 <- unique(unlist(evals[["preds"]]))
    l2 <- unique(out[["observed"]])
    
    if (all(l1 %in% l2)) {levs <- l2} # get levels in case not all show up.
    else {levs <- l1}
    
    return(confusionMatrix(data = factor(unlist(evals[["preds"]]), levels = levs), reference = factor(out[["observed"]], levels = levs), dnn = c("Predicted","Observed")))
  }
  out[["confusion"]] <- confusion(out)
  
  return(out)
}


# Evaluate with 
set.seed(0602)
sample200 <- sample(1:dim(test)[1], size = 200)
mod1_200 <- evaluate(bn_gR1, mb1, data = test[sample200,], type = "class")
mod2_200 <- evaluate(bn_gR2, mb2, data = test[sample200,], type="class")
mod1_200
mod2_200
table(test[sample200,]$vfatcount)
save.image("line209")

# Confusion Matrix (same)

save.image("line252")
load("line252")
# Scoring functions

set.seed(0602)
n <- 1000
listPreds <- list()
for (i in 1:n) {
  predictors <- sample(x = names(test)[-33], # don't choose target variable
                       size = sample(1:(dim(test)[2]-1), size = 1), replace = F)
  listPreds[[i]] <- predictors
  if (i==1) {
    evals1 <- predict.grain(bn_gR1, predictors = predictors, newdata = test[i,], type = "distribution", response = "vfatcount")
    evals2 <- predict.grain(bn_gR2, predictors = predictors, newdata = test[i,], type = "distribution", response = "vfatcount")
  }
  else {
    newRow <- predict.grain(bn_gR1, predictors = predictors, newdata = test[i,], type = "distribution", response = "vfatcount")$pred[[1]]
    evals1$pred[[1]] <- rbind(evals1$pred[[1]], newRow)
    newRow <- predict.grain(bn_gR2, predictors = predictors, newdata = test[i,], type = "distribution", response = "vfatcount")$pred[[1]]
    evals2$pred[[1]] <- rbind(evals2$pred[[1]], newRow)
  }
}


true <- test[1:n,"vfatcount"]

getdf <- function(evals) {
  df <- data.frame("true"=vector(mode="numeric", length=n), "prob"=vector(mode="numeric", length=n))
  df$true <- test[1:n, "vfatcount"]
  for (i in 1:dim(df)[1]) {
    df[i,"prob"] <- evals$pred[[1]][i,as.character(df$true[i])]
  }
  return(df)
}

df1 <- getdf(evals1)
df2 <- getdf(evals2)

sum(log(df1$prob))/dim(df1)[1]
sum(log(df2$prob))/dim(df2)[1]

# Classifications:
classes1 <- vector()
classes2 <- vector()
for (i in 1:nrow(evals1$pred$vfatcount)) {
  classes1[i] <- dimnames(evals1$pred$vfatcount)[[2]][which(max(evals1$pred$vfatcount[i,])==evals1$pred$vfatcount[i,])]
  classes2[i] <- dimnames(evals2$pred$vfatcount)[[2]][which(max(evals2$pred$vfatcount[i,])==evals2$pred$vfatcount[i,])]
}

sum(classes1==as.character(true))/n
sum(classes2==as.character(true))/n

save.image("1000preds")

# Barplot of -score for both models by predictor size (bin to groups of 5)

predictorCount <- vector()
for (i in 1:length(listPreds)) {
  predictorCount[i] <- length(listPreds[[i]])
}

DF <- cbind.data.frame(c(df1$prob, df2$prob), rep(c("1","2"), times=c(n,n)))
names(DF) <- c("prob", "model")
DF <- cbind.data.frame(DF, "num_predictors"=rep(predictorCount, times=2))

a <- seq(5,35, by=5)
b <- seq(1,34, by=5)
labs <- c(paste(b,a,sep="-"), "36-40")

DF <- transform(DF, "Bin"=cut(predictorCount, breaks = seq(0.5,40.5,by=5), labels=labs))
DF <- transform(DF, "score"=log(prob))

library(ggplot2)
gDF <- group_by(DF, Bin, model)
gDF <- summarise(gDF, "mean_score"=mean(score))

ggplot(data=gDF, aes(x=Bin, y=-mean_score, color=model)) + 
  geom_point() +
  geom_line(data=gDF, aes(x=Bin, y=-mean_score, group=model)) +
  labs(x="Number of Predictors", y="negative mean\nlogarithmic score", title="Forecast Error, -logarithmic Scoring Rule") +
  scale_color_discrete(l=40) +
  scale_y_continuous(breaks=seq(0.1, 0.7, 0.2))
  
setwd("..")
setwd("TEX presentation/")
dev.print(pdf, "slide25.pdf")

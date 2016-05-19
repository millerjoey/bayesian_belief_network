library(dplyr)
library(bnlearn)
library(gRain)

setwd("Desktop/Bayesian Data Analysis/Final Proj/")

dat <- read.table("caselisting.txt", header = T, nrows = 70000, sep = "", na.strings = ".")

# Getting more descriptive names for dat:
new.names <- c("obs", "state", "case.num", "vehicle.num",
               "person.num", "atm.cond", "num.fatalities", 
               "speeding", "age",
               "airbags", "alcohol.t.r", "alcohol.t.s",
               "alcohol.t.t", "drug.t.r", "drug.t.s", "drug.t.t",
               "injury.severity",
               "meth.alc.det", "meth.drug.det",
               "drug.involvement", "alc.involvement",
               "race",
               "seatbeltORhelmet", "sex",
               "num.fat.vehicle", "num.occupants", "travel.speed", "vehicle.make", "vehicle.model", "driver.alc.involvement",
               "driver.height.ft", "driver.height.in", "driver.weight.lb", "speed.limit")

names(dat) <- new.names

#=======Cleaning Data======
# Replace NAs with -1s:
dat <- as.data.frame(apply(X = dat, MARGIN=2, FUN = function(x) replace(x,which(is.na(x)),-1)))

# Get drivers only, eliminate useless variables
dat <- filter(dat, person.num==1)
data <- select(dat, -obs, -case.num, -vehicle.num, -person.num)
rm(dat)

# Separate data into two dfs to allow for discretize() to work:
x <- apply(X = data, MARGIN=2, FUN = function(x) length(unique(x))) > 3
data[,x] <- as.data.frame(apply(X = data[,x], MARGIN=2, FUN = function(x) x+runif(n = length(x), min = -0.001, max=0.001)))

# Hartemink method: cuts numeric data into 30 quantiles (ibreaks) and combines into 3 (breaks) in a way that reduces the mutual information
# between each variable and the other variables minimally. You will NOT get equally sized intervals in count or measure. Takes ~10s of seconds.
data[,x] <- discretize(data = data[,x], method = "hartemink", breaks = 3, ibreaks = 30, idisc="quantile")

# Turn others into factors:
data[,-x] <- as.data.frame(apply(X=data[,-x], MARGIN=2, FUN = as.factor))

#=======Split Dataset======
set.seed(666)
rand.rows <- sample(size = round(0.8*dim(data)[1]), x = (1:dim(data)[1]), replace = F)
train <- data[rand.rows,]
test <- data[-rand.rows,]

rm(data, x, new.names, rand.rows)

# # Optionally, specify a model manually
# bn.structure <- dag(
#   c("alcohol.t.r", "alcohol.t.t", "alcohol.t.s", "alc.involvement"),
#   c("alcohol.t.t", "meth.alc.det"),
#   c("speeding", "alcohol.t.r"),
#   c("seatbeltORhelmet", "airbags", "sex"),
#   c("injury.severity", "seatbeltORhelmet"),
#   c("num.fat.vehicle", "num.occupants", "speeding", "injury.severity")
# )
# 
# grain.manual <- grain(x = bn.structure, train, smooth = 1)
# grain.manual

# For the next two models, we'll only use a subset of the data. Particularly,
# the prior for the fully bayesian model is exponential in number of variables. 
# Will select 14 randomly.
set.seed(666)
cols <- sample(x = 1:30, size = 14, replace = F) # 14 variables requires a prior for the Deal package of ~100 MB.

#=====Select Variables======

# Suppose the observed and predicted (target) classes are a random 7 and 1 of the 14 in the model:
set.seed(666)
target <- sample(x = cols, size = 1, replace = F)
predictors <- sample(x = cols[-which(cols==target)], size = 7, replace = F)

# > names(train[,predictors])
# [1] "driver.height.in" "atm.cond"         "num.fatalities"   "driver.weight.lb"
# [5] "speed.limit"      "drug.t.t"         "driver.height.ft" 

# > names(train[,target, drop=F])
# [1] "drug.involvement"

# With the devil's seed, we are using the car crash data to predict drug involvement (3 classes).


# Model 2, structure learning with BIC.
# Learn a model in bnlearn
bnl.mod <- hc(x = train[,cols], score = "bic")
bnl.mod.fit <- bn.fit(bnl.mod, train[,cols], method = "bayes")

# Model 3
library(deal)
deal.net <- network(train[,cols])
prior <- jointprior(deal.net, N=10)
deal.net <- learn(deal.net, train[,cols], prior)$nw
deal.best <- autosearch(deal.net, train[,cols], prior, removecycles = T) # Prints 36 times to the console

# Export to bnlearn, which is a more complete package with many utilities for BNs
bnlearn.deal <- model2network(modelstring(deal.best$nw))
bnlearn.deal.fit <- bn.fit(bnlearn.deal, train[,cols], method = "bayes")

# Plots
graphviz.plot(bnl.mod, shape = "ellipse",highlight = list('nodes'=names(train)[c(predictors, target)], 
                                                          'fill'=c(rep('black', times=length(predictors)), "red"), 'textCol'='white', 'col'='black'))
dev.print(pdf, file="learnedBIC.pdf")

graphviz.plot(bnlearn.deal, shape = "ellipse",highlight = list('nodes'=names(train)[c(predictors, target)], 
                                                              'fill'=c(rep('black', times=length(predictors)), "red"), 'textCol'='white', 'col'='black'))
dev.print(pdf, file="deal.pdf")

# Test data:
set.seed(666)
newdata <- sample_n(test, size = 8000, replace = F)

#=======Predictions========

# Model learned via BIC score
preds1 <- predict(object = as.grain(bnl.mod.fit), predictors = names(train[,predictors]),
                  response = names(train[,target, drop=F]), newdata = newdata)
sum(unlist(preds1$pred)==as.character(newdata[,names(train[,target, drop=F])]))/dim(newdata)[1]
# 61.5%

# Fully Bayesian Model
preds2 <- predict(object = as.grain(bnlearn.deal.fit), predictors = names(train[,predictors]),
                  response = names(train[,target, drop=F]), newdata = newdata)
sum(unlist(preds2$pred)==as.character(newdata[,names(train[,target, drop=F])]))/dim(newdata)[1]
# 61.6%

# Conditional probability tables given predictor values from random entry in test dataset
set.seed(666)
observed_row <- sample_n(test, size = 1)

# Here, I use the gRain package to get conditional probability tables. as.grain() transforms models to gRain object. as.bn.fit() converts in the other direction

# Model learned via BIC.
bnl.cpt <- setFinding(as.grain(bnl.mod.fit), nodes = names(train[,predictors]), states = observed_row[,predictors])
querygrain(bnl.cpt)[names(train)[target]]               # Conditional probability table for target
querygrain(as.grain(bnl.mod.fit))[names(train)[target]] # Marginal probability table for target

# Fully Bayesian model.
deal.cpt <- setFinding(as.grain(bnlearn.deal.fit), nodes = names(train[,predictors]), states = observed_row[,predictors])
querygrain(deal.cpt)[names(train)[target]]                   # Conditional probability table for target
querygrain(as.grain(bnlearn.deal.fit))[names(train)[target]] # Marginal probability table for target

rm(list = ls())

setwd('E:/Project/Employee absentism/R')

getwd()

library("dplyr")
library("ggplot2")
library("data.table")
library('scales')
library('psych')
library('corrgram')
library('ggcorrplot')
library('rpart')
library('randomForest')
library('caret')
library('MASS')
library('e1071')
library('gbm')
library('DMwR')
library('mlr')
library('dummies')
library('DataCombine')

#loading dataset

emp = read.csv('E:/Project/Employee absentism/R/employee_absenteeism.csv',header = T)


#Checking if any NA are present
sum(is.na(emp)) #Total 135 NA values are present in dataset

colnames(emp)

str(emp)

#Here we are storing categorical and continuous variables in different objects
variable_num = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
                    'Work.load.Average.day', 'Transportation.expense',
                    'Hit.target', 'Weight', 'Height','Son', 'Pet', 
                    'Body.mass.index', 'Absenteeism.time.in.hours')

variable_cat = c('ID','Reason.for.absence','Month.of.absence','Day.of.the.week',
                     'Seasons','Disciplinary.failure', 'Education', 'Social.drinker',
                     'Social.smoker')

##################################Missing Values Analysis###############################################
emp_missing_val = data.frame(apply(emp,2,function(x){sum(is.na(x))}))

emp_missing_val$Columns = row.names(emp_missing_val)

names(emp_missing_val)[1] =  "Missing_percentage"

emp_missing_val$Missing_percentage = (emp_missing_val$Missing_percentage/nrow(emp)) * 100

emp_missing_val = emp_missing_val[order(-emp_missing_val$Missing_percentage),]

row.names(emp_missing_val) = NULL

emp_missing_val = emp_missing_val[,c(2,1)]

write.csv(emp_missing_val, "Missing_perc.csv", row.names = F)

ggplot(data = emp_missing_val[1:21,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
     geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
     ggtitle("Missing data percentage (emp)") + theme_bw() + scale_x_discrete(labels = abbreviate)


#actual value at 11th row 6th column = 260
#Mean=220.98
#Median = 225
#KNN = 260 ## here k was 3

#Creating NA
#emp[11,6] = NA

#Mean method 
#emp$Transportation.expense[is.na(emp$Transportation.expense)]= mean(emp$Transportation.expense,na.rm = T)

#Creating NA
#emp[11,6] = NA

#Median
#emp$Transportation.expense[is.na(emp$Transportation.expense)]=median(emp$Transportation.expense,na.rm = T)


#Creating NA
emp[11,6] = NA

#KNN method
emp = knnImputation(emp, k = 3)

sum(is.na(emp))


##############  Outlier Analysis   ###################
# Boxplot for continuous variables
for (i in 1:length(variable_num))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (variable_num[i]), x = "Absenteeism.time.in.hours"), data = subset(emp))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=variable_num[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box plot of absenteeism for",variable_num[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)
gridExtra::grid.arrange(gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,ncol=2)



#Replace all outliers with NA and impute
# #create NA on "Transportation.expense"


for(i in variable_num)
{
  val = emp[,i][emp[,i] %in% boxplot.stats(emp[,i])$out]
  #print(length(val))
  emp[,i][emp[,i] %in% val] = NA
}

sum(is.na(emp))

# Imputing missing values
emp = knnImputation(emp,k=3)

sum(is.na(emp))

################### Feature Selection   ########################
## Correlation Plot 
ggcorrplot(cor(emp[,variable_num]),method = 'square', lab = T, title =  'Correlation plot',
           ggtheme = theme_dark())

## ANOVA test for Categprical variable

summary(aov(formula = Absenteeism.time.in.hours~ID,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Reason.for.absence,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Month.of.absence,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Day.of.the.week,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Seasons,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Disciplinary.failure,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Education,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Social.drinker,data = emp))
summary(aov(formula = Absenteeism.time.in.hours~Social.smoker,data = emp))


##### Dimension Reduction ######
## Here we are dropping continous variable who have high correlation and categorical variable who have
## p value greater than 0.05.
emp_reduced = subset(emp, select = -c(Weight,Month.of.absence,Age))


#Since we have dropped few variables and stored in emp_reduced. So here we have to updated the variable
#and store them in new object.
variable_num_update = c('Distance.from.Residence.to.Work', 'Service.time',
                        'Work.load.Average.day', 'Transportation.expense',
                        'Hit.target','Height','Son', 'Pet', 
                        'Body.mass.index', 'Absenteeism.time.in.hours')

variable_cat_update = c('ID','Reason.for.absence','Day.of.the.week',
                        'Seasons','Disciplinary.failure', 'Education', 'Social.drinker',
                        'Social.smoker')


############################     Feature Scaling    ############################################
#Normality check
qqnorm(emp_reduced$Absenteeism.time.in.hours)
hist(emp_reduced$Transportation.expense)

######### Normalisation
for(i in variable_num_update){
  print(i)
  emp_reduced[,i] = (emp_reduced[,i] - min(emp_reduced[,i]))/
    (max(emp_reduced[,i] - min(emp_reduced[,i])))
}

# Creating dummy variables for categorical variables

emp_dummy = dummy.data.frame(emp_reduced, variable_cat_update)

###### Checking VIF to see if there is multicollinearity
#install.packages('usdm')
library(usdm)
vif(emp_reduced[,-18])  ## It will calculate vif

vifcor(emp_reduced[,-18], th = 0.9)




########################    Model Development   ##############################
#Clean the environment
rmExcept("emp_dummy")

#divide data into train and test. Here we have used simple random sampling since our target variable is continous
train_index = sample(1:nrow(emp_dummy) , .80*nrow(emp_dummy))

#Divide data into trin and test
train = emp_dummy[train_index,]
test = emp_dummy[-train_index,]

######################  Dimensionality Reduction using PCA   ########################


#principal component analysis
pc_analysis = prcomp(train)

#compute standard deviation of each principal component
std_dev = pc_analysis$sdev

#compute variance
pca_var = std_dev^2

#proportion of variance explained
prop_varex = pca_var/sum(pca_var)

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#add a training set with principal components
train.data = data.frame(Absenteeism.time.in.hours = train$Absenteeism.time.in.hours, pc_analysis$x)

# From the above plot selecting 50 components since they are explaining almost 95% of variance
train.data =train.data[,1:50]

#transform test into PCA
test.data = predict(pc_analysis, newdata = test)
test.data = as.data.frame(test.data)

#select the first 50 components
test.data=test.data[,1:50]

###################  Decision Tree  #########################


#Develop Model on training data
fit_DT = rpart(Absenteeism.time.in.hours ~., data = train.data, method = "anova")


#Lets predict for training data
pred_DT_train = predict(fit_DT, train.data)

#Lets predict for training data
pred_DT_test = predict(fit_DT,test.data)


# For training data 
print(postResample(pred = pred_DT_train, obs = train$Absenteeism.time.in.hours))

#   RMSE        Rsquared        MAE 
# 0.10718208    0.73612633     0.07558583 

# For testing data 
print(postResample(pred = pred_DT_test, obs = test$Absenteeism.time.in.hours))

#    RMSE        Rsquared         MAE 
# 0.1580666     0.3681254      0.1139894 


####################   Linear Regression     ############################

#Develop Model on training data
fit_LR = lm(Absenteeism.time.in.hours ~ ., data = train.data)

#Lets predict for training data
pred_LR_train = predict(fit_LR, train.data)

#Lets predict for testing data
pred_LR_test = predict(fit_LR,test.data)

# For training data 
print(postResample(pred = pred_LR_train, obs = train$Absenteeism.time.in.hours))

#   RMSE           Rsquared         MAE 
# 0.0190320        0.9909747    0.0115488 

# For testing data 
print(postResample(pred = pred_LR_test, obs =test$Absenteeism.time.in.hours))
 
#    RMSE        Rsquared        MAE 
# 0.02526831    0.98757496    0.01434393 

################################   Random Forest    ##################################

#Develop Model on training data
fit_RF = randomForest(Absenteeism.time.in.hours~.,importance = TRUE,ntree = 500, data = train.data)

#Lets predict for training data
pred_RF_train = predict(fit_RF, train.data)

#Lets predict for testing data
pred_RF_test = predict(fit_RF,test.data)

# For training data 
print(postResample(pred = pred_RF_train, obs = train$Absenteeism.time.in.hours))

#    RMSE         Rsquared           MAE 
# 0.04471945     0.98042601      0.03280806 

# For testing data 
print(postResample(pred = pred_RF_test, obs = test$Absenteeism.time.in.hours))

#    RMSE          Rsquared          MAE 
# 0.1192099       0.7717611        0.0835930 



####Linear Regression model performs better so we will go with linear regression model.

# for training data the values are

#   RMSE           Rsquared         MAE 
# 0.0190320        0.9909747    0.0115488 

#for testing data the values are

#    RMSE        Rsquared        MAE 
# 0.02526831    0.98757496    0.01434393 



















































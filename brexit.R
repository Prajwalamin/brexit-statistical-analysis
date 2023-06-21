
# installing packages
install.packages("tidyverse")
install.packages("caTools")    # For Logistic regression
install.packages("ROCR")       # For ROC curve to evaluate model
install.packages('rpart')
install.packages('rpart.plot')
install.packages('caret')
install.packages('carData')
install.packages('dplyr')

# using packages
library(tidyverse)
library(ggplot2)
library(caTools)
library(ROCR) 
library(rpart)
library(rpart.plot)
library(caret)
library(car)
library(carData)
library(dplyr)

#Loading the file
brexit <- read.csv("brexit.csv")


str(brexit)
head(brexit)
view(brexit)

brexit$voteBrexit <- as.numeric(brexit$voteBrexit)
nrow(brexit)

# preprocessing


#Splitting the data
set.seed(2)
split <- sample.split(brexit, SplitRatio = 0.8)
split
train <- subset(brexit, split == "TRUE")
test <- subset(brexit, split == "FALSE")
nrow(train)

# fitting data to the model
model <- glm(voteBrexit ~ ., family = binomial, data = train)
summary(model)
vif(model)


# Ordering the coefficients based on their magnitudes
summary_coef <- summary(model)$coefficients

#predict

preds <- predict.glm(model, test, type = c("response"))
view(prediction_probs)
view(pred)

#convert predictions to 1 or 0
prediction_probs <- ifelse(pred > 0.5, 1, 0)

#EVALUATION
#Create a confusion matrix

confusionMatrix(table(test$voteBrexit, prediction_probs))

#ROC-AUC Curve
ROCPred <- prediction(prediction_probs, test$voteBrexit) 
ROCPer <- performance(ROCPred, measure = "tpr", 
                      x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc # first model: 0.8132, second model: 0.8198
ROCPer
# Plotting curve
plot(ROCPer)
plot(ROCPer, colorize = TRUE, 
     print.cutoffs.at = seq(0.1, by = 0.1), 
     main = "ROC CURVE")
abline(a = 0, b = 1)

auc <- round(auc, 4)
legend(.6, .4, auc, title = "AUC", cex = 1)

cor(test$withHigherEd, pred)


# Plotting the predictions 
plot(test$withHigherEd, test$voteBrexit, xlab = "Higher Educated Individuals")
lines(smooth.spline(brexit$withHigherEd, prediction_probs), col = "blue", lwd=2)

ggplot(test, aes(x=pred, y=medianIncome, color = voteBrexit)) + 
  geom_point(alpha=.7, stroke = 1, size=1.5) +
  geom_line(aes(x = 0.5, y = 0.5)) +
  scale_color_gradient(low = "#ffee00", high = "#00b3ff") +
  theme_classic() +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) + # set fixed axis ranges
  geom_vline(xintercept = 0.5, color = "gray", alpha = 0.5) + # add vertical line at x-mean
  geom_hline(yintercept = 0.5, color = "gray", alpha = 0.25) # add horizontal line at y-mean
  
theme(panel.grid.major = element_blank(), # remove major grid lines
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.border = element_blank(), # remove panel border
        panel.background = element_rect(fill = "white"), # set background color
        plot.background = element_rect(fill = "white"))


plot(brexit$withHigherEd, pred)

# Extract the coefficients from the model (COEFFICIENT ORDERING)
summary_coef <- summary(model)$coefficients
view(summary_coef)
# Sort the coefficients by magnitude
sorted_coefficients <- sort(abs(coefficients), decreasing = TRUE)

# Print the coefficients in descending order
for (i in order(sorted_coefficients)) {
  print(paste(names(coefficients)[i], round(coefficients[i], 4)))
}

ggplot(test, aes(x=pred, y=withHigherEd, color = voteBrexit)) + 
  geom_point(alpha=.9) 

  
ggplot(brexit, aes(x=medianIncome, y=withHigherEd, color = voteBrexit)) + 
  geom_point(alpha=.7) +
  theme_minimal() 
  stat_smooth(method="glm", se=FALSE, method.args = list(family=binomial))
  
round(5.8385, 2)


# <--- TASK 3 --->

brexit_reduced <- select(brexit, -abc1)


#Splitting the data into training & testing (80:20)
set.seed(2)

train_reduced <- subset(brexit_reduced, split == "TRUE")
test_reduced <- subset(brexit_reduced, split == "FALSE")
nrow(train_reduced)

# Training the model
model_new <- glm(voteBrexit ~ ., data = train_reduced, family = "binomial")
summary(model_new)

# Predictions
predictions_new <- predict(model_new, newdata = test_reduced, type = "response")

# Convert predictions to 1 or 0
prediction_conv <- ifelse(predictions_new > 0.5, 1, 0)

#Create a confusion matrix
cm2 <- confusionMatrix(table(test_reduced$voteBrexit, prediction_conv))
cm2$overall["Accuracy"]


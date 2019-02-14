# Section 1
# read in cars data
cars = read.csv("http://www.rob-mcculloch.org/data/susedcars.csv")

# access cars headers as variables
attach(cars)

# plot mileage vs price
plot(mileage, price, xlab="Mileage", ylab="Price")

#fit linear model to price and mileage
linearFit = lm(price ~ mileage, cars)

# plot linear fit
abline(linearFit$coef, col="red", lwd=2)

# import kNN library
library(kknn)

# isolate relevant data
train = data.frame(mileage, price)
test = data.frame(mileage = sort(mileage))

# run kNN with k = 50
# NOTE: this is NOT the best k value. I just chose something to get everything working
k50 = kknn(price~mileage, train, test, k=50, kernel = "optimal")

#add knn fit to plot
lines(test$mileage, k50$fitted.values, col="blue", lwd=2)

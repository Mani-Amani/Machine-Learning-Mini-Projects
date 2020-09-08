#mini_Script that calculates the mean absolute error between the given data and the predicted data
#X is a given model from a previous code snippet

#finding out the mean absolue error
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

#implementing a tree regressor
from sklearn.tree import DecisionTreeRegressor
hs_model = DecisionTreeRegressor(random_state=1)
hs_model.fit(X, y)

#using the ML model to fit our prediction and given data to calculate the MAE
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
hs_model = DecisionTreeRegressor()
# Fit model
hs_model.fit(train_X, train_y)
val_predictions = hs_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))



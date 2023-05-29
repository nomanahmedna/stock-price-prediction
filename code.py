import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the CSV file
df = pd.read_csv('stock_prices.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['OPEN', 'HIGH', 'LOW']], df['CLOSE'], test_size=0.25)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print('Mean squared error:', mse)

# Predict the next price
next_price = model.predict(df[['OPEN', 'HIGH', 'LOW']].tail(1))

# Print the predicted next price
print('Predicted next price:', next_price)

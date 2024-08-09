import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Grab the Data from CSV
melbourne_data = pd.read_csv('./data/melb_data.csv')

# Drop NA to do very basic clean up
melbourne_data = melbourne_data.dropna(axis=0)

# Select which features to use
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Select the target
y = melbourne_data['Price']

# Train/Test split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Create a model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit the model
melbourne_model.fit(train_X, train_y)

# Make predictions
predictions = melbourne_model.predict(val_X)

# Calculate MAE
mae = mean_absolute_error(val_y, predictions)
print(f'Mean Absolute Error: {mae}')
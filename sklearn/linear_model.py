import pandas as pd
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def linear_regression(features, target) -> tuple[lm.LinearRegression, r2_score]:
    """
    This function takes the feature data, target data, 
    and it creates a linear regression model, and it's r2 score.
    Args:
        features: Pandas Dataframe
        target: Pandas Dataframe
    Returns:
        LinearRegression model, r2 score
    """

    # Train/Test split the data
    train_X, val_X, train_y, val_y = train_test_split(features, target, random_state=0)

    # Create a model
    model = lm.LinearRegression()

    # Fit the model
    model.fit(train_X, train_y)

    # Make predictions
    predictions = model.predict(val_X)

    # Calculate R Squared 
    r_square = r2_score(val_y, predictions)
    
    return model, r_square


# Grab the Data from CSV
fuel_data = pd.read_csv('./data/fuel_consumption.csv')
cols = fuel_data.columns

# Select which features to use
cylinders = ['Cylinders']
X = fuel_data[cylinders]

# Select the target
y = fuel_data['CO2 emissions (g/km)']

# Get the LinearRegression model
model, r_square = linear_regression(features=X, target=y)

print(f"RandomForestRegressor model has an R Squared of {r_square}")

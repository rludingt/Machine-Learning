import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def random_forest_regressor(features, target) -> tuple[RandomForestRegressor, mean_absolute_error]:
    """
    This function takes the feature data, target data,
    and it trains a RandomForestRegressor model. 
    The function will return the model, and it's MAE.

    Args:
        features: Pandas Dataframe
        target: Pandas Dataframe
    Returns:
        RandomForestRegressor model, Mean Absolue Error (MAE)
    """

    # Train/Test split the data
    train_X, val_X, train_y, val_y = train_test_split(features, target, random_state=0)

    # Create a model
    model = RandomForestRegressor(random_state=1)

    # Fit the model
    model.fit(train_X, train_y)

    # Make predictions
    predictions = model.predict(val_X)

    # Calculate MAE
    mae = mean_absolute_error(val_y, predictions)
    
    return model, mae

### EXAMPLE ###

# Grab the Data from CSV
melbourne_data = pd.read_csv('./data/melb_data.csv')

# Drop NA to do very basic clean up
melbourne_data = melbourne_data.dropna(axis=0)

# Select which features to use
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Select the target
y = melbourne_data['Price']

# Get the RandomForestRegressor model
model, mae = random_forest_regressor(features=X, target=y)

print(f"RandomForestRegressor model has an MAE of {mae}")

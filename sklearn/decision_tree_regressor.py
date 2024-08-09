import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def decision_tree_regressor(features, target, max_leaf_nodes) -> tuple[DecisionTreeRegressor, int]:
    """
    This function takes the feature data, target data, and a list of max_leaf_node integers,
    and it trains a DecisionTreeRegressor model. The function will return the model, and it's MAE.

    Args:
        features: Pandas Dataframe
        target: Pandas Dataframe
        max_leaf_nodes: int
    Returns:
        DecisionTreeRegressor model, Mean Absolue Error (MAE)
    """

    # Train/Test split the data
    train_X, val_X, train_y, val_y = train_test_split(features, target, random_state=0)

    # Create a model
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

    # Fit the model
    model.fit(train_X, train_y)

    # Make predictions
    predictions = model.predict(val_X)

    # Calculate MAE
    mae = mean_absolute_error(val_y, predictions)
    
    return model, mae


def leaf_node_tuner(features, target, max_leaf_node_list) -> DecisionTreeRegressor:
    """
    This function takes the feature data, target data, and a list of max_leaf_node integers,
    and it trains a DecisionTreeRegressor model with each max_leaf_node integer param. 
    The function will then print the MAE, and return the model with the best performance.

    Args:
        features: Pandas Dataframe
        target: Pandas Dataframe
        max_leaf_node_list: List of integers
    Returns:
        DecisionTreeRegressor model
    """

    # Test all leaf_node_list integers
    results_dict = {}
    for leaf in max_leaf_node_list:
        model, mae = decision_tree_regressor(features, target, leaf)
        results_dict[mae] = model
        print(f"A max_leaf_count of {leaf} produced MAE {mae}")

    # Grab the lowest MAE
    mae_list = list(results_dict.keys())
    best_mae = min(mae_list)
    best_model = results_dict[best_mae]

    return best_model


### EXAMPLE ###


# Grab the Data from CSV
melbourne_data = pd.read_csv('./data/melb_data.csv')

# Drop NA to do very basic clean up
melbourne_data = melbourne_data.dropna(axis=0)

# Select which features to use
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Select the target
y = melbourne_data['Price']

# Test a few max_leaf_node integer options to tune and find the best model
max_leaf_node_list = [5, 50, 500, 5000]
model = leaf_node_tuner(features=X, target=y, max_leaf_node_list=max_leaf_node_list)
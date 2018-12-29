from utils.extract_features import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.extract_features import *
from sklearn.metrics import f1_score
from sklearn import svm
from joblib import dump
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

preprocess_all = False
E_F = features_dataset(preprocess_all)
if preprocess_all:
    E_F.prepocess_train_data()
    E_F.prepocess_t.std(axis=0)


train_feature, training_labels, test_feature = E_F.load_features_all()

# Training and Testing Sets
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(train_feature, training_labels,
                                                                            test_size = 0.25, random_state = 42)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint



def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [5,7,9,11,13,14],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [ 300,500, 700, 1000]
}

# Create a base model
rf = RandomForestRegressor(random_state = 42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)



# Fit the grid search to the data
grid_search.fit(train_features, train_labels);

grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)




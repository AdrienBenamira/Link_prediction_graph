from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.extract_features import *
from sklearn.metrics import f1_score
from sklearn import svm
from joblib import dump
import lightgbm as lgb



preprocess_all = False
norm =True

E_F = features_dataset(preprocess_all)
if preprocess_all:
    E_F.prepocess_data()


train_features, training_labels, test_features = E_F.load_features_all()
m = train_features.mean(axis=0)
std = train_features.std(axis=0)

if norm:
    train_features = (train_features - m) / std
    test_features = (test_features - m) / std


"""
a_tester = [0,1,2,4,5]
train_features = train_features[:, a_tester]
test_features = test_features[:, a_tester]
"""


def train_predict_save(model, name):
    fscore_t = f1_score(training_labels[:int(train_features.shape[0] * 0.5)],
                        model.predict(train_features[:int(train_features.shape[0] * 0.5)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0] * 0.5):],
                        model.predict(train_features[int(train_features.shape[0] * 0.5):]))
    print(name +" model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))
    dump(model, './model/model'+str(name)+ str(fscore_v) + '.joblib')
    predictions = model.predict(test_features)
    df = pd.DataFrame(predictions)
    df.columns = ["category"]
    df.to_csv(name + '.csv')



from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search
param_grid = {
    'penalty': ['l2'],
    'dual': [True, False],
    'tol': [1e-4,1e-3,1e-5],
    'C': [0.1, 1, 10],
}

# Create a base model
rf = svm.LinearSVC()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)



# Fit the grid search to the data
grid_search.fit(train_features, training_labels);
grid_search.best_params_
best_grid = grid_search.best_estimator_
train_predict_save(best_grid, "predictions_Grid")




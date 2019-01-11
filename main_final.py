from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from utils.extract_features import *
from sklearn import svm
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from utils import Config



config = Config("config/")

E_F = features_dataset(config.preprocess_all)
if config.preprocess_all:
    E_F.prepocess_data()

train_features, training_labels, test_features = E_F.load_features_all()

m = train_features.mean(axis=0)
std = train_features.std(axis=0)

if config.norm:
    train_features = (train_features - m) / std
    test_features = (test_features - m) / std

a_tester = config.features_a_tester

if config.model == "GB":
    model = lgb.LGBMClassifier(objective='binary', reg_lambda=config.reg_lambda_gb,
                               n_estimators=config.n_estimator_GB   )
if config.model =="RF":
    model = RandomForestClassifier(n_estimators=500)
if config.model == "LinSVM":
    model = svm.LinearSVC()
if config.model == "Lin":
    model = LogisticRegression()

train_features = train_features[:, a_tester]
test_features = test_features[:, a_tester]

kf = StratifiedKFold(n_splits=config.num_split_cross_val, shuffle=True)

train_features = pd.DataFrame(train_features)
training_labels = pd.DataFrame(training_labels)

predicts = []
for train_index, test_index in kf.split(train_features, training_labels):
    n = int((1.0-config.pourcentage_split_val)*(len(train_index)+len(test_index)))
    random.shuffle(test_index)
    test_index_new = test_index[:n]
    train_index_new = np.union1d(test_index[n:], train_index)
    X_train, X_val = train_features.iloc[train_index_new], train_features.iloc[test_index_new]
    y_train, y_val = training_labels.iloc[train_index_new], training_labels.iloc[test_index_new]
    if config.model == "GB":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            early_stopping_rounds=50)
    else:
        model.fit(X_train, y_train)
    predicts.append(model.predict(test_features))

predict_by_hh = pd.DataFrame(np.array(predicts).mean(axis=0).round().astype(int),
                             columns=['category'])
predict_by_hh.to_csv(config.name_output)

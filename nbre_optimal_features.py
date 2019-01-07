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


train_features, training_labels, test_features = E_F.load_features_all()


m = train_features.mean(axis=0)
std = train_features.std(axis=0)


train_features = (train_features - m) / std
test_features = (test_features - m) / std

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV



# Create the RFE object and compute a cross-validated score.
svc = RandomForestClassifier(n_estimators=500)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='f1')
rfecv.fit(train_features, training_labels)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
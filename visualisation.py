from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.extract_features import *
from sklearn.metrics import f1_score
from sklearn import svm
from joblib import dump
import lightgbm as lgb
import matplotlib.pyplot as plt


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


dataframe = pd.DataFrame(train_features)

dataframe.hist()
plt.show()



correlations = dataframe.corr().abs()

s = correlations.unstack()
so = s.sort_values(kind="quicksort")

print (so)

# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=0, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()

pd.plotting.scatter_matrix(dataframe, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.show()
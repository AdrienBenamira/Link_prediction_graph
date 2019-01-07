from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.extract_features import *
from sklearn.metrics import f1_score
from sklearn import svm
from joblib import dump
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression




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



#0 overlap_title,
#1 temp_diff,
#2  comm_auth,
#3 num_inc_edges,
#4  Distance_abstract,
#5  Distance_title,
#6 shortest_path_dijkstra
#7 shortest_path_dijkstra_und
#8 ,comm_neighbors,
#9 no_edge,
#10  tfidf_distance_corpus,
#11  tfidf_distance_titles,
#12 jaccard_und
#13 Resource_allocation

#Test 1 : Glove 83%>LDA-IDF 81%


a_tester = [1,4,6,7,8,10,11]
train_features = train_features[:, a_tester]
test_features = test_features[:, a_tester]




def train_predict_save(model, name):
    fscore_t = f1_score(training_labels[:int(train_features.shape[0] * 0.99)],
                        model.predict(train_features[:int(train_features.shape[0] * 0.99)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0] * 0.99):],
                        model.predict(train_features[int(train_features.shape[0] * 0.99):]))
    print(name +" model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))
    dump(model, './model/model'+str(name)+ str(fscore_v) + '.joblib')
    predictions = model.predict(test_features)
    df = pd.DataFrame(predictions)
    df.columns = ["category"]
    df.to_csv(name + '.csv')


modelGB = lgb.LGBMClassifier(objective='binary', reg_lambda=10, n_estimators=10000)
modelGB.fit(train_features[:int(train_features.shape[0] * 0.99)],
                training_labels[:int(train_features.shape[0] * 0.99)],
                eval_set=[(train_features[int(train_features.shape[0] * 0.99):],
                           training_labels[int(train_features.shape[0] * 0.99):])],
                early_stopping_rounds=50, verbose=False)
modelRF = RandomForestClassifier(n_estimators=500)
modelRF.fit(train_features[:int(train_features.shape[0]*0.99)], training_labels[:int(train_features.shape[0]*0.99)])
modelSVM = svm.LinearSVC()
modelSVM.fit(train_features[:int(train_features.shape[0]*0.99)], training_labels[:int(train_features.shape[0]*0.99)])
logit_model = LogisticRegression()
logit_model.fit(train_features[:int(train_features.shape[0]*0.99)], training_labels[:int(train_features.shape[0]*0.99)])

train_predict_save(modelRF, "predictions_RF")
train_predict_save(modelSVM, "predictions_SVM")
train_predict_save(modelGB, "predictions_GB")
train_predict_save(logit_model, "predictions_linear")


"""
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
Cs = [0.001, 0.01, 0.99, 1, 10]
gammas = [0.001, 0.01, 0.99, 1]
kernels = ['linear', 'rbf', 'poly']
param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : kernels}
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=3,  n_jobs = -1, verbose = 2)
grid_search.fit(train_features, training_labels)
best_grid = grid_search.best_estimator_
train_predict_save(best_grid, "predictions_Grid")
"""


import numpy as np
import matplotlib.pyplot as plt

importances = modelRF.feature_importances_
std = np.std([tree.feature_importances_ for tree in modelRF.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_features.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train_features.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train_features.shape[1]), indices)
plt.xlim([-1, train_features.shape[1]])
plt.show()


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
    fscore_t = f1_score(training_labels[:int(train_features.shape[0]*0.5)],
                        model.predict(X_new[:int(train_features.shape[0]*0.5)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0]*0.5):],
                        model.predict(X_new[int(train_features.shape[0]*0.5):]))
    print(name +" model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))
    dump(model, './model/model'+str(name)+ str(fscore_v) + '.joblib')

    predictions = model.predict(X_test_new)
    df = pd.DataFrame(predictions)
    df.columns = ["category"]
    df.to_csv(name + '.csv')



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

model = SelectKBest(f_classif, k=8).fit(train_features, training_labels)
X_new = model.transform(train_features)
X_test_new = model.transform(test_features)

modelRF = RandomForestClassifier(n_estimators=500)
modelRF.fit(X_new[:int(train_features.shape[0]*0.5)], training_labels[:int(train_features.shape[0]*0.5)])


train_predict_save(modelRF, "predictions_RF_select_fetautres")





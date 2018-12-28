from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.extract_features import *
from sklearn.metrics import f1_score
from sklearn import svm
from joblib import dump
import lightgbm as lgb


preprocess_all = False
norm =False

E_F = features_dataset(preprocess_all)
if preprocess_all:
    E_F.prepocess_train_data()
    E_F.prepocess_t.std(axis=0)


train_features, training_labels, test_features = E_F.load_features_all()
m = train_features.mean(axis=0)
std = train_features.std(axis=0)

if norm:
    train_features = (train_features - m) / std
    test_features = (test_features - m) / std



def train_predict_save(model, name):
    fscore_t = f1_score(training_labels[:int(train_features.shape[0] * 0.9)],
                        model.predict(train_features[:int(train_features.shape[0] * 0.9)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0] * 0.9):],
                        model.predict(train_features[int(train_features.shape[0] * 0.9):]))
    print(name +" model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))
    dump(modelSVM, './model/modelSVM' + str(fscore_v) + '.joblib')
    predictions = model.predict(test_features)

    if 0.001<fscore_v<0.1:
        predictions[predictions == 1] = 2
        predictions[predictions == 0] = 1
        predictions[predictions == 2] = 0

    df = pd.DataFrame(predictions)
    df.columns = ["category"]
    df.to_csv(name + '.csv')



for k in range(20):
    modelSVM = svm.LinearSVC()
    modelSVM.fit(train_features[:int(train_features.shape[0]*0.9)], training_labels[:int(train_features.shape[0]*0.9)])
    train_predict_save(modelSVM, "predictions_SVM")
    train_features = (train_features - m) / std
    test_features = (test_features - m) / std
    modelSVM_norm = svm.LinearSVC()
    modelSVM_norm.fit(train_features[:int(train_features.shape[0] * 0.9)],
                 training_labels[:int(train_features.shape[0] * 0.9)])
    train_predict_save(modelSVM_norm, "predictions_SVM")
    train_features = (train_features + m) * std
    test_features = (test_features + m) * std
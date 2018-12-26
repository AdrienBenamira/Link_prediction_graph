from utils.extract_features import *
from sklearn.metrics import f1_score
from sklearn import svm
from joblib import dump



preprocess_all = False
E_F = features_dataset(preprocess_all)

if preprocess_all:
    E_F.prepocess_train_data()
    E_F.prepocess_test_data()


train_features, training_labels, test_features = E_F.load_features_all()


for i in range(20):
    """
    modelGB = lgb.LGBMClassifier(objective='binary', reg_lambda=10, n_estimators=10000)
    modelGB.fit(train_features[:int(train_features.shape[0]*0.9)], training_labels[:int(train_features.shape[0]*0.9)],
        eval_set=[(train_features[int(train_features.shape[0]*0.9):], training_labels[int(train_features.shape[0]*0.9):])],
        early_stopping_rounds=50, verbose=False)

    fscore_t = f1_score(training_labels[:int(train_features.shape[0]*0.9)], modelGB.predict(train_features[:int(train_features.shape[0]*0.9)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0]*0.9):], modelGB.predict(train_features[int(train_features.shape[0]*0.9):]))
    print("Gradboost model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))

    modelRF = RandomForestClassifier(n_estimators=500)
    modelRF.fit(train_features[:int(train_features.shape[0]*0.9)], training_labels[:int(train_features.shape[0]*0.9)])
    fscore_t = f1_score(training_labels[:int(train_features.shape[0]*0.9)], modelRF.predict(train_features[:int(train_features.shape[0]*0.9)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0]*0.9):], modelRF.predict(train_features[int(train_features.shape[0]*0.9):]))
    print("Random Forest model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))
    """
    modelSVM = svm.LinearSVC()
    modelSVM.fit(train_features[:int(train_features.shape[0]*0.9)], training_labels[:int(train_features.shape[0]*0.9)])
    fscore_t = f1_score(training_labels[:int(train_features.shape[0]*0.9)], modelSVM.predict(train_features[:int(train_features.shape[0]*0.9)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0]*0.9):], modelSVM.predict(train_features[int(train_features.shape[0]*0.9):]))
    print("SVM model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))
    dump(modelSVM, './model/modelSVM'+str(fscore_v)+'.joblib')
    predictions = modelSVM.predict(test_features)
    """
    predictions[predictions == 0] = 2
    predictions[predictions == 1] = 0
    predictions[predictions == 2] = 1
    pd.DataFrame(predictions).to_csv('predictions.csv')
    """
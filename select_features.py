from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from utils.extract_features import *
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
training_labels = training_labels[:int(train_features.shape[0] * 0.1)]
train_features = train_features[:int(train_features.shape[0] * 0.1)]

print(train_features.shape)
print(training_labels.shape)


fetures_test = np.linspace(train_features.shape[1],1,train_features.shape[1])-1

def combinliste(seq, k):
    p = []
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                s.append(seq[j])
            j += 1
        if len(s)==k:
            p.append(s)
        i += 1
    return p


def train_predict_save(model, name):
    fscore_t = f1_score(training_labels[:int(train_features.shape[0] * 0.1)],
                        model.predict(X_test[:int(train_features.shape[0] * 0.1)]))
    fscore_v = f1_score(training_labels[int(train_features.shape[0] * 0.1):],
                        model.predict(X_test[int(train_features.shape[0] * 0.1):]))
    print(name + " model: F1 score - Training %.3f - Validation %.3f" % (fscore_t, fscore_v))

for k in range(9, 1, -1):
    a_tester = combinliste(fetures_test, k+2)
    for i, liste in enumerate(a_tester):
        liste = list(map(int, liste))
        X_test = train_features[:, liste]
        #clf = svm.LinearSVC()
        #modelRF = RandomForestClassifier(n_estimators=500)
        #scores = cross_val_score(clf, X_test, training_labels,cv=5, scoring='f1_macro')
        #scoresRF = cross_val_score(modelRF, X_test, training_labels, cv=3, scoring='f1_macro')
        modelSVM = svm.LinearSVC()
        scoresSVM = modelSVM.fit(X_test, training_labels)
        logit_model = LogisticRegression()
        logit_model = logit_model.fit(X_test, training_labels)

        modelGB = lgb.LGBMClassifier(objective='binary', reg_lambda=10, n_estimators=10000)
        modelGB.fit(X_test[:int(train_features.shape[0] * 0.1)],
                    training_labels[:int(train_features.shape[0] * 0.1)],
                    eval_set=[(X_test[int(train_features.shape[0] * 0.1):],
                               training_labels[int(train_features.shape[0] * 0.1):])],
                    early_stopping_rounds=50, verbose=False)

        train_predict_save(modelSVM, "SVM")
        train_predict_save(modelGB, "GB")
        print(liste)



        """
        fscore_t = f1_score(training_labels[:int(train_features.shape[0] * 0.9)],
                            clf.predict(X_test[:int(train_features.shape[0] * 0.9)]))
        fscore_v = f1_score(training_labels[int(train_features.shape[0] * 0.9):],
                            clf.predict(X_test[int(train_features.shape[0] * 0.9):]))
        """

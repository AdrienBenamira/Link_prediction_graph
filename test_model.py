from joblib import load
import os
from utils.extract_features import *
import pandas as pd




preprocess_all = False
E_F = features_dataset(preprocess_all)

if preprocess_all:
    E_F.prepocess_train_data()
    E_F.prepocess_test_data()


train_features, training_labels, test_features = E_F.load_features_all()

for root, dirs, files in os.walk("./model"):
    for file in files:
        if file.endswith(".joblib"):
            clf = load(os.path.join(root, file))
            predictions = clf.predict(test_features)

            print(file.split('.')[1])

            if file.split('.')[1]<0.5:
                predictions[predictions == 0] = 2
                predictions[predictions == 1] = 0
                predictions[predictions == 2] = 1

            df = pd.DataFrame(predictions)
            df.columns = ["category"]
            df.to_csv(file+'.csv')

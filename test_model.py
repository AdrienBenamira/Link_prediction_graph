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


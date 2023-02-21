import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from ml_logic.params import filename

def define_train_model():
    model = LogisticRegression(max_iter=1000)
    return model

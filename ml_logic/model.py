import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from ml_logic.params import filename
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def define_train_model():
    model = RandomForestClassifier(max_depth=20, n_estimators=2000, n_jobs=-1)
    return model

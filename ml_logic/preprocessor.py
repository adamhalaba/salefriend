from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from ml_logic.params import filename_2
import pickle

# Creation of basic pipeline

def pipeline_preproc_basic():

    r_scaler = RobustScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    imputer = SimpleImputer(strategy='most_frequent')

    num_transformer = make_pipeline(imputer, r_scaler)
    num_col = make_column_selector(dtype_include=['int64','float64'])

    cat_transformer = make_pipeline(imputer, ohe)
    cat_col = ['country_name',
                'request_date_day',
                'month_request',
                'day_time_request',
                'country_name_us',
                'civility']


    preproc_basic = make_column_transformer(
                    (num_transformer,num_col),
                    (cat_transformer,cat_col),
    #                (custom_transformer, custom_col),
                    remainder='passthrough')
    #if X.shape[0]==1:
        #pipeline_1 = preproc_basic.transform(X)
    #else:
    #pipeline_1 =preproc_basic.fit_transform(X)
    #pipeline_1 = pd.DataFrame(pipeline_1)

    return preproc_basic

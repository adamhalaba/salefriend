import random
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

def civility_replacer(serie):
    # Replace values of 'Dr.' and 'Prof.' with NaN
    values_to_replace = ['Dr.', 'Prof.']
    serie.replace(values_to_replace, np.nan, inplace=True)

    # Calculate the ratio of female to male
    female_count = serie[serie['civility'] == 'Ms'].shape[0]
    male_count = serie[serie['civility'] == 'Mr'].shape[0]
    ratio = round(female_count / (female_count + male_count), 1)

    # Replace NaN values with 'Ms' or 'Mr' using the calculated ratio
    nan_indexes = serie[serie.isna()].index
    replacement = random.choices(['Ms', 'Mr'], weights=[ratio, (1-ratio)], k=len(nan_indexes))
    serie.loc[nan_indexes, 'civility'] = replacement
    #merged_df['civility'].loc[nan_indexes, 'civility'] = replacement
    return serie

# Creation of basic pipeline

def make_pipeline_fit_transform(X: pd.DataFrame):

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
                'country_name_us']

    civility = FunctionTransformer(lambda x: civility_replacer(x))

    custom_transformer = make_pipeline(civility, ohe)
    custom_col = ['civility']

    preproc_basic = make_column_transformer(
                    (num_transformer,num_col),
                    (cat_transformer,cat_col),
                    (custom_transformer, custom_col),
                    remainder='passthrough')
    if X.shape[0]==1:
        pipeline_1 = preproc_basic.transform(X)
    else:
        pipeline_1 =preproc_basic.fit_transform(X)

    pipeline_1 = pd.DataFrame(pipeline_1)

    return pipeline_1

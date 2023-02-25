
import numpy as np
import pandas as pd
import os
import pickle
import sklearn

from ml_logic.data import clean_data
from ml_logic.params import LOCAL_DATA_PATH
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from ml_logic.preprocessor import pipeline_preproc_basic
from ml_logic.model import define_train_model
from ml_logic.params import filename
from sklearn.model_selection import cross_val_score, cross_validate

def preprocess_and_train():
    # Load Datasets
    df_requests = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Requests.csv"))
    df_bookings = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Bookings.csv"))
    df_users = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Users.csv"))
    df_logins = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Logins.csv"))

    #Return final Dataset
    data = clean_data(df_requests, df_bookings, df_users, df_logins)
    X_new = data.loc[[0]]
    X_new = X_new.drop(columns=["booked"])
    #print(data.shape)
    #print(X_new)

    # Balancing
    data_1 = data[data['booked'] == 1]
    data_0 = data[data['booked'] == 0]\
            .sample(n=round(len(data_1)*1.5))
    data_undersampled = pd.concat([data_0, data_1], axis=0).sample(frac=1)
    X = data_undersampled.drop(columns='booked')
    y = data_undersampled['booked']



    #data =

    # Create X, y
    # X = data.drop(columns=['booked'])
    # y = data['booked']

    # Define Preprocess num, cat and custom
    preproc_basic = pipeline_preproc_basic()

    # Define model (?)
    model = define_train_model()

    preproc_full = make_pipeline(preproc_basic, model)

    full_model = preproc_full.fit(X,y)

    # save the model to disk -> not working for now
    with open('model.pickle', 'wb') as f:
        pickle.dump(full_model, f)
    #filename = 'finalized_model.sav'
    #pickle.dump(full_model, open(filename, 'wb'))

    return None

def predict(X_new: pd.DataFrame = None):

    if X_new is None:

        X_new = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],# useless but the pipeline requires it
            is_new_user=[0],
            id_charter_company=[232],
            charter_type=[0],
            country_name=['Spain'],
            destination_flexible=[0],
            flexible_date=[0],
            request_date_day=['Sunday'],
            month_request=['April'],
            day_time_request=['evening'],
            days_before_departure=[60],
            in_europe=[1],
            num_passengers=[2],
            kid_on_board=[0],
            duration=[7],
            civility=['Mr'],
            country_name_us=['Spain'],
            is_mac=[1.0]
        ))
    #load trained model
    with open('model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)


    #return probability to be booked, ie. booked=1
    #Predict X_new (y_pred)
    y_pred = loaded_model.predict(X_new)

    #Probability of being booked, ie. y_pred=1
    y_proba_booked= loaded_model.predict_proba(X_new)
    print(y_proba_booked)
    print(f'({y_pred}, Probability of conversion:{round(y_proba_booked[0][1],3)*100} %)')
    return y_pred


if __name__== "__main__":
    #preprocess_and_train()
    predict()
    print("End")

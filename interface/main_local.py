
import numpy as np
import pandas as pd
import os
import pickle


from ml_logic.data import clean_data
from ml_logic.params import LOCAL_DATA_PATH
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from ml_logic.preprocessor import pipeline_preproc_basic
from ml_logic.model import define_train_model
from ml_logic.params import filename

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
    #data =

    # Create X, y
    X = data.drop(columns=['booked'])
    y = data['booked']

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

    #only for test prediction for X_new
    # X_new = pd.DataFrame(dict(
    #         key=["2013-07-06 17:18:00"],# useless but the pipeline requires it
    #         id_user=[298572],
    #         is_new_user=[0],
    #         id_charter_company=[300],
    #         charter_type=[1],
    #         country_name=['Thailand'],
    #         destination_flexible=[0],
    #         flexible_date=[7],
    #         request_date_day=['Friday'],
    #         month_request=['December'],
    #         day_time_request=['night'],
    #         days_before_departure=[168],
    #         in_europe=[0],
    #         num_passengers=[5],
    #         kid_on_board=[0],
    #         duration=[6],
    #         civility=['Mr'],
    #         country_name_us=['Singapore'],
    #         is_mac=[1.0]
    #     ))

    # #Predict X_new (y_pred)
    # y_pred = full_model.predict(X_new)

    # #Probability of being booked, ie. y_pred=1
    # y_proba_booked= full_model.predict_proba(X_new)
    # print(f'({y_pred}, {round(y_proba_booked[0][1],2)})')
    return None

def predict(X_new: pd.DataFrame = None):

    if X_new is None:

        X_new = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],# useless but the pipeline requires it
            id_user=[298572],
            is_new_user=[0],
            id_charter_company=[300],
            charter_type=[1],
            country_name=['Thailand'],
            destination_flexible=[0],
            flexible_date=[7],
            request_date_day=['Friday'],
            month_request=['December'],
            day_time_request=['night'],
            days_before_departure=[168],
            in_europe=[0],
            num_passengers=[5],
            kid_on_board=[0],
            duration=[6],
            civility=['Mr'],
            country_name_us=['Singapore'],
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
    print(f'({y_pred}, {round(y_proba_booked[0][1],2)})')
    return y_pred


if __name__== "__main__":
    preprocess_and_train()
    predict()
    #predict()
    # Load Datasets
    # df_requests = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Requests.csv"))
    # df_bookings = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Bookings.csv"))
    # df_users = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Users.csv"))
    # df_logins = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Logins.csv"))

    #Return final Dataset
    # data = clean_data(df_requests, df_bookings, df_users, df_logins)
    # X_new = data.loc[[0]]
    # X_new = X_new.drop(columns=["booked"])
    # predict(X_new)
    print("End")


import numpy as np
import pandas as pd
import os
import pickle
from ml_logic.data import clean_data
from ml_logic.params import LOCAL_DATA_PATH
from sklearn.model_selection import train_test_split
from ml_logic.preprocessor import make_pipeline_fit_transform
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
    print(data.shape)
    print(X_new)

    # Balancing
    #data =

    # Create X, y
    X = data.drop(columns=['booked'])
    y = data['booked']


    # Preprocess X using proprocessor.py
    X_processed = make_pipeline_fit_transform(X)

    # Train model on X_processed and y, using `model.py`
    model = define_train_model(X_processed,y)
    X_new_processed = make_pipeline_fit_transform(X_new)
    loaded_model = pickle.load(open(filename,"rb"))
    return model.predict(X_new_processed)

    #return model

#def predict(X_new):
    X_new_processed = preprocess_features(X_new)
    loaded_model = pickle.load(open(filename,"rb"))
    return loaded_model.predict(X_new_processed)



if __name__== "__main__":
    preprocess_and_train()
    print("test")

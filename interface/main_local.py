
import numpy as np
import pandas as pd
import os
from ml_logic.data import clean_data
from ml_logic.params import LOCAL_DATA_PATH

def preprocess_and_train():
    # Load Datasets
    df_requests = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Requests.csv"))
    df_bookings = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Bookings.csv"))
    df_users = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Users.csv"))
    df_logins = pd.read_csv(os.path.join(LOCAL_DATA_PATH,"Logins.csv"))

    #Return final Dataset
    data = clean_data(df_requests, df_bookings, df_users, df_logins)
    return data

if __name__== "__main__":
    df= preprocess_and_train()
    print(df.head(5))

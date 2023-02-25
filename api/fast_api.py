#from fastapi import File
from fastapi import FastAPI
import pickle
import pandas as pd
import pydantic
from interface.main_local import pred

app = FastAPI()

class PredictionForm(pydantic.BaseModel):
    is_new_user : float
    country_name_us : str
    civility: str
    is_mac : float
    num_passengers : float
    kid_on_board : float
    in_europe : float
    duration : float
    destination_flexible : float
    country_name : str
    flexible_date : float
    id_charter_company : float
    charter_type : float
    request_date_day : str
    month_request : str
    day_time_request : str
    days_before_departure : float

def load_model():
    #load trained model
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

@app.get("/")
def read_root():
    return {"message": "Congrats!!! API is working"}


@app.post("/predict")
def predict(input_features: PredictionForm):

    input_features = input_features.dict()
    #print("Input",input_features)
    X_new = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],
            is_new_user=[input_features['is_new_user']],
            id_charter_company=[input_features['id_charter_company']],
            charter_type=[input_features['charter_type']],
            country_name=[input_features['country_name']],
            destination_flexible=[input_features['destination_flexible']],
            flexible_date=[input_features['flexible_date']],
            request_date_day=[input_features['request_date_day']],
            month_request=[input_features['month_request']],
            day_time_request=[input_features['day_time_request']],
            days_before_departure=[input_features['days_before_departure']],
            in_europe=[input_features['in_europe']],
            num_passengers=[input_features['num_passengers']],
            kid_on_board=[input_features['kid_on_board']],
            duration=[input_features['duration']],
            civility=[input_features['civility']],
            country_name_us=[input_features['country_name_us']],
            is_mac=[input_features['is_mac']]
        ))
    # TO BE CLEANED LATER
    #model = load_model()
    #result = model.predict_proba(X_new)
    #result = round(result[0][1],3)*100

    # "pred" function returns the probability of being booked (y=1) with 3 decimals
    result = pred(X_new)
    # Convert to percentage
    result = result*100
    print("Model Testing = ", result)
    return {"Probability of Booking": result, "Features" : input_features}

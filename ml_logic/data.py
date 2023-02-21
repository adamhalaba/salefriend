import pandas as pd
import numpy as np

from ml_logic.preprocessor import civility_replacer

def clean_data(df_requests_originals: pd.DataFrame,df_bookings_originals: pd.DataFrame, df_users_originals: pd.DataFrame, df_logins: pd.DataFrame):
    #Preselection de colonnes


    df_requests = df_requests_originals.loc[:,['id_request', 'id_user', 'is_new_user', 'request_date',
       'id_charter_company',  'charter_type',
       'country_name', 'destination_flexible',  'adults',
       'kids', 'start_date', 'end_date', 'flexible_date']]

    df_bookings = df_bookings_originals.loc[:,['id_booking', 'id_user', 'country_end_name', 'start_date', 'end_date']]

    df_users = df_users_originals.loc[:,['id_user','creation_date', 'civility', 'country_name']]

    #Feature Creation
    #convert date columns from objects to datetime64
    df_bookings['start_date']=pd.to_datetime(df_bookings['start_date'], format="%m/%d/%Y")
    df_bookings['end_date']=pd.to_datetime(df_bookings['end_date'], format="%m/%d/%Y")

    #convert date columns from objects to datetime64
    df_requests['start_date']=pd.to_datetime(df_requests['start_date'], format="%m/%d/%Y")
    df_requests['end_date']=pd.to_datetime(df_requests['end_date'], format="%m/%d/%Y")
    df_requests['request_date']=pd.to_datetime(df_requests['request_date'], format="%m/%d/%Y %H:%M") #"%d/%m/%y %H:%M"

    #   1. A Creation of request_date_new (DS -df_requests)
    df_requests["request_date_new"]=pd.to_datetime(df_requests["request_date"]).dt.date

    # 2. Creation of request_day_day (DS -df_requests)
    df_requests["request_date_day"]=pd.to_datetime(df_requests["request_date"]).dt.day_name()

    # 3. Creation of month_request (DS -df_requests)
    df_requests['month_request'] = pd.to_datetime(df_requests['request_date']).dt.month_name()

    # 4. Creation of day_time_request (DS -df_requests)
    df_requests['hour_request'] = pd.to_datetime(df_requests['request_date']).dt.hour
    df_requests.loc[df_requests['hour_request'] <= 6, "day_time_request"] = "night"
    df_requests.loc[(df_requests['hour_request'] > 6) & (df_requests['hour_request'] <= 12), "day_time_request"] = "morning"
    df_requests.loc[(df_requests['hour_request'] > 12) & (df_requests['hour_request'] <= 18), "day_time_request"] = "afternoon"
    df_requests.loc[df_requests['hour_request'] >= 18, "day_time_request"] = "evening"

    # 5. Creation of days_before_departure (DS -df_requests)
    df_requests["days_before_departure"]=((df_requests["start_date"]-df_requests["request_date"])/np.timedelta64(1,"D")).astype(np.int64)

    # 6. Creation of in_europe (DS -df_requests)
    df_requests["country_lower_case"]=df_requests["country_name"].str.lower()
    european_countries = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom']
    lowercase_countries = [country.lower() for country in european_countries]
    df_requests["in_europe"]=df_requests["country_lower_case"].isin(lowercase_countries).astype(int)

    # 7.  Creation of num_passengers (DS -df_requests)
    df_requests['num_passengers']=df_requests['adults']+df_requests['kids']

    # 8. Creation of kid_on_board (DS -df_requests)
    df_requests['kid_on_board'] = df_requests['kids'].apply(lambda x: 1 if x > 0 else 0)

    # 9. Creation of duration (DS -df_requests)
    df_requests["duration"]=((df_requests["end_date"]-df_requests["start_date"])/np.timedelta64(1,"D")).astype(np.int64)

    # 10. Create of is_MAC and nb_of_logs
    df_logins['is_mac']=df_logins['user_agent'].apply(get_mac)
    mac_log_df = df_logins.groupby('id_user').agg({'is_mac':'mean','log_date':'count'}).rename(columns={"log_date": "nb_of_logs"})
    common_device = lambda x: 0 if x<0.5 else 1
    mac_log_df['is_mac']= mac_log_df['is_mac'].apply(common_device)

    #merge left to keep all requests and only bookings with requests
    merged_df= df_requests.merge(df_bookings,on='id_user', how='left', suffixes=['_rq', '_bk'])
    merged_df= merged_df.merge(df_users,on='id_user',how='left', suffixes=['','_us'])
    #Merge mac_log_df to merged_df and add the suffix lg (for log)
    merged_df= merged_df.merge(mac_log_df,on='id_user',how='left', suffixes=['','_lg'])

    #Feature creation
    # Creation of seniority_of_client
    merged_df["creation_date"].replace("0000-00-00 00:00:00","7/19/2009 10:54",inplace=True)
    merged_df["creation_date"]=pd.to_datetime(merged_df['creation_date'], format="%m/%d/%Y %H:%M")
    merged_df["seniority_of_client"]=(merged_df["request_date"]-merged_df["creation_date"]).dt.days

    # Creation of month_depart
    merged_df['month_depart'] = pd.to_datetime(merged_df['start_date_rq']).dt.month_name()

    # Create Target
    # Calculate the difference between the request's start date and booking's start date
    merged_df['date_diff'] = np.abs(pd.to_datetime(merged_df['start_date_rq']) - pd.to_datetime(merged_df['start_date_bk']))

    # Check if the difference is smaller than a certain threshold
    #threshold = pd.Timedelta('14 days')

    # Calculate the difference between the request's start date and booking's start date
    merged_df['date_diff'] = np.abs(pd.to_datetime(merged_df['start_date_rq']) - pd.to_datetime(merged_df['start_date_bk']))

    # Check if the difference is smaller than a certain threshold
    threshold = pd.Timedelta('14 days')

    # Create new column 'booked' 1: request with booking, 0: request without booking or request with datediff > 14 days
    merged_df['booked'] = np.where((merged_df['id_booking'].isna()) |  (merged_df['date_diff'] > threshold) , 0, 1)

    # Drop duplicated requests
    merged_df = merged_df.sort_values(by='date_diff', ascending=True).drop_duplicates(subset='id_request', keep='first')

    # Re-order
    merged_df = merged_df.sort_values(by='id_request', ascending=True)

    #Outliers

    # delete rowns with days_before_departure values with bigger than 365*2=730 days
    merged_df.drop(merged_df[merged_df["days_before_departure"] > 730].index, inplace = True)

    # delete rows with duration values bigger than 28 days
    merged_df.drop(merged_df[merged_df["duration"] > 28].index, inplace = True)

    # correct civility nan, prof, dr

    merged_df['civility'] = civility_replacer(pd.DataFrame(merged_df['civility']))

    data = merged_df[['id_user', 'is_new_user', 'id_charter_company', 'charter_type', 'country_name',
       'destination_flexible', 'flexible_date', 'request_date_day',
       'month_request', 'day_time_request',
       'days_before_departure',  'in_europe',
       'num_passengers', 'kid_on_board', 'duration', 'civility',
       'country_name_us', 'is_mac', 'booked']]

    return data



def get_mac(user_agent):
    mac_list = ['Macintosh', 'iPhone','iPad', 'iPod']
    for mac in mac_list:
        if mac in user_agent:
            return 1
        return 0

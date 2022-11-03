import numpy
import pandas as pd
import json
from datetime import datetime
import pickle

# flatten json
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], a + name)
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

# load static data
def load_static():
    static_df = pd.read_csv(".\\data\\static\\staticFeatures.csv", sep=';', encoding='utf-8', index_col=None)
    static_df.drop('Unnamed: 10', axis='columns', inplace=True)
    static_df['timestamp'] = pd.to_datetime(static_df['timestamp'], infer_datetime_format=True)
    static_df.set_index('timestamp', inplace=True)
    return static_df

# load weather data
def load_weather():
    with open(".\\data\\weather\\converted.json") as f:
        lines = f.read().splitlines()

    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    for j in range(df_final.shape[0]):
        print("\r" + str(j), end="")
        for i in range(49):
            if i != 0:
                df_final['hourly.data'][j][i].pop("time", None)
            df_final['hourly.data'][j][i].pop("icon", None)
            df_final['hourly.data'][j][i].pop("apparentTemperature", None)
            df_final['hourly.data'][j][i].pop("uvIndex", None)
            df_final['hourly.data'][j][i].pop("precipType", None)
            df_final['hourly.data'][j][i].pop("summary", None)
            df_final['hourly.data'][j][i].pop("precipIntensity", None)
            df_final['hourly.data'][j][i].pop("precipProbability", None)
            """
            df_final['hourly.data'][0][i]["temperature" + str(i)] = df_final['hourly.data'][0][i].pop("temperature")
            df_final['hourly.data'][0][i]["dewPoint" + str(i)] = df_final['hourly.data'][0][i].pop("dewPoint")
            df_final['hourly.data'][0][i]["humidity" + str(i)] = df_final['hourly.data'][0][i].pop("humidity")
            df_final['hourly.data'][0][i]["pressure" + str(i)] = df_final['hourly.data'][0][i].pop("pressure")
            df_final['hourly.data'][0][i]["windSpeed" + str(i)] = df_final['hourly.data'][0][i].pop("windSpeed")
            df_final['hourly.data'][0][i]["windBearing" + str(i)] = df_final['hourly.data'][0][i].pop("windBearing")
            df_final['hourly.data'][0][i]["cloudCover" + str(i)] = df_final['hourly.data'][0][i].pop("cloudCover", 0.0)
            df_final['hourly.data'][0][i]["visibility" + str(i)] = df_final['hourly.data'][0][i].pop("visibility")
            """

        df_final['hourly.data'][j] = flatten_json(df_final['hourly.data'][j])

    weather_df = pd.json_normalize(df_final['hourly.data'])

    # remove ozone and windGust
    columns = list(weather_df.columns)
    for a in columns:
        if a.find("ozone") != -1:
            weather_df.pop(a)
        if a.find("windGust") != -1:
            weather_df.pop(a)

    # fill in missing values
    weather_df.interpolate(axis=0, limit=None, inplace=True)
    weather_df.fillna(method="backfill", inplace=True)
    weather_df.fillna(method="ffill", inplace=True)

    weather_df['timestamp'] = pd.to_datetime(weather_df['time0']*1000*1000*1000)
    foo = weather_df.pop('time0')
    weather_df.set_index('timestamp', inplace=True)

    # remove duplicates
    weather_df = weather_df.loc[~weather_df.index.duplicated(keep='first')]

    return weather_df

# load sensor data
def load_sensor(n):
    sensor_df = pd.read_json('./data/smartmeters/N' + str(n) + '.json')
    sensor_df.pop("_id")
    sensor_df.pop("node_id")
    sensor_df.pop("stamp_db")

    try:
        sensor_df.pop("pg")
    except:
        pass

    try:
        sensor_df.pop("qc")
    except:
        pass

    try:
        sensor_df.pop("qg")
    except:
        pass

    sensor_df['timestamp'] = pd.to_datetime(sensor_df['stamp'] * 1000 * 1000 * 1000) # unix ts in nanoseconds
    foo = sensor_df.pop("stamp")
    sensor_df.set_index('timestamp', inplace=True)

    # enrich sensor_df
    # add 15, 30, 45-minute values
    sensor_df["pc15"] = sensor_df.shift(periods=-1, freq='0.25H')["pc"]
    sensor_df["pc30"] = sensor_df.shift(periods=-2, freq='0.25H')["pc"]
    sensor_df["pc45"] = sensor_df.shift(periods=-3, freq='0.25H')["pc"]
    sensor_df = sensor_df[:-3]

    # add historic values
    moving_averages = ['1H', '6H', '1D', '7D', '30D']
    delays = ['1H', '2H', '3H', '6H', '12H', '1D', '2D', '3D', '7D']

    # create moving averages
    for a in moving_averages:
        sensor_df['pc_ma_' + a] = sensor_df['pc'].rolling(window=a, min_periods=1).mean()
        sensor_df['pc_std_' + a] = sensor_df['pc'].rolling(window=a, min_periods=1).std()

        # historic values
    for a in moving_averages:
        for d in delays:
            sensor_df["pc_ma_{}_{}".format(a, d)] = sensor_df.shift(periods=1, freq=d)["pc_ma_{}".format(a)]

    return sensor_df

# make final expects static and weather data to be loaded
def make_final(static_df, weather_df, n):
    sensor_df = load_sensor(n)
    final_df = pd.concat([sensor_df, static_df], axis=1, join="inner")
    final2_df = pd.concat([weather_df, final_df], axis=1, join="inner")
    return final2_df.dropna()


# loading features (features.pkl is produced by 04_Checking_Common_Features.ipynb)
def load_features():
    with open("results/features.pkl", "rb") as f:
        features = pickle.load(f)
    return(features)

# features includes a list of feature-set candidates, grouped by sensor and horizon
# we want to 
#   (1) extract the best features per horizon (for direct modelling)
#   (2) extract a common featureset per sensor (for structured modelling)

def get_candidates(s, h):
    features = load_features()
    for candidate in features:
        if (candidate[0]["sensor"] == s) and (candidate[0]["horizon"] == h):
            return candidate
    
    
def get_features_sh(s, h):
    candidates = get_candidates(s, h)
    accuracy = -999
    i = -1
    c = 0
    for candidate in candidates:
        if candidate["accuracy"] > accuracy:
            accuracy = candidate["accuracy"]
            i = c
            c = c + 1
        
    return candidates[i]["features"]

def normalize_accuracies(horizon):
    # normalize accuracies
    accuracies = []
    for candidate in horizon:
        accuracies.append(candidate["accuracy"])        
    a_max = max(accuracies)
    a_min = min(accuracies)
    
    if a_min > a_max - 0.2:
        a_min = a_max - 0.2
    k = 1 / (a_max - a_min)
    
    for i in range(len(accuracies)):        
        new_acc = (accuracies[i] - a_min) * k
        horizon[i]["accuracy"] = new_acc        
    
    return(horizon)
    
def get_all_candidates(s):
    features = load_features()
    # build a normalized accuracies candidate list
    candidates = []
    for candidate in features:
        if (candidate[0]["sensor"] == s):
            candidates.append(normalize_accuracies(candidate))
    
    # build a list of best candidate features
    fweights = {}
    for horizon in candidates:
        for candidate in horizon:
            new_acc = candidate["accuracy"]
            for f in candidate["features"]:
                #print(f, new_acc)
                if not f in fweights:
                    fweights[f] = 0
                fweights[f] = fweights[f] + new_acc
    
    # filter feature candidates
    useful_features = []
    for f in fweights:
        v = fweights[f]
        if v >= 1.0:
            useful_features.append(f)
    
    return useful_features
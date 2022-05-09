import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations
    global __data_columns

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    print("loading model...start")
    global __model

    with open("./artifacts/Bangeluru_Home_Price_Prediction.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("loading artifacts..done")

def get_estimated_price(location, sqft, bath, bedrooms):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bedrooms

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("1st phase jp nagar",1000,2,2))

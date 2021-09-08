import json
import pickle
import jsonify
import numpy as np


__locations = None
__model = None
__data_columns = None

def get_estimated_price(location,sqft,bhk,bath):
    try: 
        loc_index=__data_columns.index(location.lower())
        
    except:
        loc_index=-1
        
    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1

    return round(__model.predict([x])[0],2)



def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "rb")as f:
        __data_columns=json.load(f)['data-columns']
        __locations=__data_columns[3:]

    global __model
    if __model is None:
        with open('./artifacts/home_prices_model.pickle', 'rb')as f:
            __model=pickle.load(f)

    print('Loading saved artifacts...done')


def get_locations_names():
    return __locations

def get_data_columns():
    return __data_columns



if __name__=="__main__":
    load_saved_artifacts()
    print(get_locations_names())
    # print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
    # print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
    # print(get_estimated_price('Kalhalli',1000,2,2))
    # print(get_estimated_price('Ejipura',1000,2,2))

        
    
import dill
import json
import pandas as pd
import glob
from datetime import datetime
import os

path = os.environ.get('PROJECT_PATH', '.')


def predict():

    files_model = os.listdir(f'{path}/data/models/')
    files_model = [os.path.join(f'{path}/data/models/', file_model) for file_model in files_model]
    files_model = [file_model for file_model in files_model if os.path.isfile(file_model)]
    last_model = max(files_model, key=os.path.getctime)
    with open(last_model, 'rb') as file:
        model = dill.load(file)

    files = glob.glob(f'{path}/data/test/*.json')
    combined = pd.DataFrame()

    for filee in files:

        with open(filee, "r", encoding="utf-8") as fil:
            form = json.load(fil)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)

            result_dict = {

                "id": form["id"],
                "Result": y[0],
                "price": form["price"]

            }

            data_frame = pd.DataFrame([result_dict])
            combined = pd.concat([combined, data_frame])

    combined.to_csv(f'{path}/data/predictions/all_result{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()

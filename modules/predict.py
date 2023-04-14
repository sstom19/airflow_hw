import logging
import json
import glob
import os
import dill
import pandas as pd


path = os.environ.get('PROJECT_PATH', '.')


def predict():
    # получаем список доступных моделей и выбираем созданную последней
    model_names = glob.glob(f'{path}/data/models/*.pkl')
    model_names.sort(reverse=True)
    model_name = model_names[0]
    timestamp = model_name[-16:-4]
    with open(model_name, 'rb') as file:
        model = dill.load(file)
    logging.info(f'Loaded model from file {model_name}')

    # цикл по json файлам в каталоге test
    list_files = glob.glob(f'{path}/data/test/*.json')
    predicts = []
    for file_name in list_files:
        with open(file_name) as file:
            json_data = json.load(file)
            df = pd.DataFrame.from_dict([json_data])
            y = model.predict(df)
            predicts.append({'car_id': df['id'][0], 'pred': y[0]})
            logging.info(f'Predicted value for file {file_name}')
    result_filename = f'{path}/data/predictions/preds_' + timestamp + '.csv'
    pd.DataFrame(predicts).to_csv(result_filename, index=False)
    logging.info(f'Predictions saved to file {result_filename}')


if __name__ == '__main__':
    predict()

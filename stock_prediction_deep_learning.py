# Copyright 2020-2022 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import secrets
import pandas as pd
import argparse
from datetime import datetime, timedelta
import numpy as np

from stock_prediction_class import StockPrediction
from stock_prediction_lstm import LongShortTermMemory
from stock_prediction_numpy import StockData
from stock_prediction_plotter import Plotter
from stock_prediction_readme_generator import ReadmeGenerator

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def train_LSTM_network(stock):
    data = StockData(stock)
    plotter = Plotter(True, stock.get_project_folder(), data.get_stock_short_name(), data.get_stock_currency(), stock.get_ticker())
    (x_train, y_train), (x_test, y_test), (training_data, test_data, all_data) = data.download_transform_to_numpy(stock.get_time_steps(), stock.get_project_folder())
    plotter.plot_histogram_data_split(training_data, test_data, stock.get_validation_date())

    lstm = LongShortTermMemory(stock.get_project_folder())
    model = lstm.create_model(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_defined_metrics())
    history = model.fit(x_train, y_train, epochs=stock.get_epochs(), batch_size=stock.get_batch_size(), validation_data=(x_test, y_test),
                        callbacks=[lstm.get_callback()])
    print("saving weights")
    model.save(os.path.join(stock.get_project_folder(), 'model_weights.h5'))

    plotter.plot_loss(history)
    plotter.plot_mse(history)

    print("display the content of the model")
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    print("plotting prediction results")
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = data.get_min_max().inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
    test_predictions_baseline.to_csv(os.path.join(stock.get_project_folder(), 'predictions.csv'))

    test_predictions_baseline.rename(columns={0: stock.get_ticker() + '_predicted'}, inplace=True)
    test_predictions_baseline = test_predictions_baseline.round(decimals=0)
    test_predictions_baseline.index = test_data.index
    plotter.project_plot_predictions(test_predictions_baseline, test_data)

    generator = ReadmeGenerator(stock.get_github_url(), stock.get_project_folder(), data.get_stock_short_name())
    generator.write()

    print("prediction is finished")

    return (data, model, plotter, all_data)

def append_new_price(input_df, input_new_price, input_current_date):
    weekday = input_current_date.weekday()
    next_date = input_current_date

    if weekday == 4:
        next_date = input_current_date + timedelta(days=3)
    elif weekday == 5:
        next_date = input_current_date + timedelta(days=2)
    else:
        next_date = input_current_date + timedelta(days=1)

    next_date = next_date.replace(hour=0, minute=0, second=0, microsecond=0)
    input_df.loc[next_date] = input_new_price
    return next_date

def predict_values(input_stock, input_data, input_model, input_plotter, input_time_steps, input_df, input_date, input_predict_size):

    for predict_index in range(0,  input_predict_size):
        # inputs = input_df[input_df.shape[0] - time_steps:]
        # test_scaled = input_data.get_min_max().fit_transform(inputs)
        inputs = input_df[(-1*input_time_steps):].values
        # print(inputs)
        test_scaled = input_data.get_min_max().transform(inputs)
        # Testing Data Transformation
        x_test = []

        x_test.append(test_scaled[0 : input_time_steps])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        test_predictions_baseline = input_model.predict(x_test)
        test_predictions_baseline = input_data.get_min_max().inverse_transform(test_predictions_baseline)
        input_date = append_new_price(input_df, test_predictions_baseline[0], input_date)


    input_df.to_csv(os.path.join(input_stock.get_project_folder(), 'predictions_' +str(input_predict_size)+'.csv'))

    input_df = input_df.round(decimals=0)
    input_plotter.project_plot_predictions_only(input_df, input_predict_size)




# The Main function requires 3 major variables
# 1) Ticker => defines the short code of a stock
# 2) Start date => Date when we want to start using the data for training, usually the first data point of the stock
# 3) Validation date => Date when we want to start partitioning our data from training to validation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("parsing arguments"))
    #parser.add_argument("-ticker", default="^FTSE")
    parser.add_argument("-ticker", default="^IXIC")
    parser.add_argument("-start_date", default="1900-01-01")
    parser.add_argument("-validation_date", default="2021-09-01")
    # parser.add_argument("-epochs", default="170")
    parser.add_argument("-epochs", default="1")
    parser.add_argument("-batch_size", default="50")
    # parser.add_argument("-time_steps", default="3")
    parser.add_argument("-predict_size", default="60")
    parser.add_argument("-time_steps", default="3")
    parser.add_argument("-github_url", default="https://github.com/jungsoo827/stock-prediction-2/raw/master/")

    args = parser.parse_args()

    PREDICT_SIZE = int(args.predict_size)
    STOCK_TICKER = args.ticker
    STOCK_START_DATE = pd.to_datetime(args.start_date)
    STOCK_VALIDATION_DATE = pd.to_datetime(args.validation_date)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    TIME_STEPS = int(args.time_steps)
    TODAY_RUN = datetime.today().strftime("%Y%m%d")

    GITHUB_URL = args.github_url
    print('Ticker: ' + STOCK_TICKER)
    print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))




    def report(input_secret, input_stock_symbol):

        TOKEN = input_stock_symbol + '_' + TODAY_RUN + '_' + input_secret
        print('Test Run Folder: ' + TOKEN)
        folder = "report/" + TODAY_RUN + '_' + input_secret + "/" + input_stock_symbol

        # create project run folder
        PROJECT_FOLDER = os.path.join(os.getcwd(), folder)
        if not os.path.exists(PROJECT_FOLDER):
            os.makedirs(PROJECT_FOLDER)

        stock_prediction = StockPrediction(input_stock_symbol,
                                           STOCK_START_DATE,
                                           STOCK_VALIDATION_DATE,
                                           PROJECT_FOLDER,
                                           GITHUB_URL,
                                           EPOCHS,
                                           TIME_STEPS,
                                           TOKEN,
                                           BATCH_SIZE)
        # Execute Deep Learning model
        (data, model, plotter, all_data) = train_LSTM_network(stock_prediction)


        last_data_date = all_data.index.values[-1]
        last_data_date = pd.Timestamp(last_data_date).to_pydatetime()
        last_data_date = last_data_date.replace(hour=0, minute=0, second=0, microsecond=0)
        next_date = last_data_date + timedelta(days=1)
        next_date = next_date.replace(hour=0, minute=0, second=0, microsecond=0)
        current_date = datetime.today()
        current_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        print(last_data_date, current_date, next_date)
        if next_date == current_date:
            current_date = current_date - timedelta(days=1)
        print(current_date)
        predict_values(stock_prediction, data, model, plotter, TIME_STEPS, all_data, current_date, PREDICT_SIZE)


    secret_token = secrets.token_hex(16)
    report(secret_token, '^IXIC')  # NASDAQ
    report(secret_token, 'AI') # C3.ai
    report(secret_token, 'NVDA') # NVIDIA
    report(secret_token, 'ACAD') # Acadia Pharmaceuticals
    report(secret_token, 'TSLA') # Tesla
    report(secret_token, 'AMZN') # Amazon
    report(secret_token, 'AAPL') # Apple
    report(secret_token, 'MSFT') # Microsoft
    report(secret_token, 'GOOG') # Goole
    report(secret_token, 'KO') # Coca Cola
    report(secret_token, 'BTC-USD') # bitcoin
    report(secret_token, '051900.KS') # lg생활건강
    report(secret_token, '090430.KS') # 아모레퍼시픽
    report(secret_token, '095700.KQ') # 제넥신
    report(secret_token, '002310.KS') # 아세아제지

    # soundhound data starts at 2022-04-28
    STOCK_VALIDATION_DATE = pd.to_datetime("2023-04-01")
    report(secret_token, 'SOUN') # Sound Hound

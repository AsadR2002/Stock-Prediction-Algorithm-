from flask import Flask
from flask import request
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


app = Flask(__name__)

@app.route("/")
def index():
    covidResponse = request.args.get("covidResponse", "")
    if covidResponse:
        answer = generate_answer(covidResponse)
    else:
        answer = ""
    return (
            """
            <head>
             <title>Ontario COVID Chatbot</title>
            </head>
            <br>
            <h1 class="mainTitle" >Ontario's COVID Help AI Chatbot</h1>
            <hr STYLE="background-color:#000000; height:5px; width:95%;">
            <form action="" method="get" class="formPrompt" autocomplete="off">
                Ask Question about Covid Protocols, Restrictions, and other General COVID Information:
                 <br>
                 <input type="text" name="covidResponse" class ="inputText">
                <input type="submit" value="Ask Question" class ="submitButton">
            </form>    
            <h1 class="answerText">Answer: </h1>
                """
            + '''<h1 style = "color: black; margin: 25px 50px 10px 50px; font-size: 1.5rem; font-weight: 700; font-family: Oswald, sans-serif !important;">''' + answer + '''</h1>'''
    )

def generate_answer(covidResponse):
    try:
        company = 'FB'

        start = dt.datetime(2012, 1, 1)
        end = dt.datetime(2020, 1, 1)

        data = web.DataReader(company, 'yahoo', start, end)

        # Prepare Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        prediction_days = 60

        x_train = []
        y_train = []

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the Model
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # prediction of the next closing value

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32)

        test_start = dt.datetime(2020, 1, 1)
        test_end = dt.datetime.now()

        test_data = web.DataReader(company, 'yahoo', test_start, test_end)
        actual_prices = test_data['Close'].values

        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs + 1), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    except ValueError:
        return "invalid input"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
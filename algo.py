import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time
import yfinance as yf
from get_all_tickers import get_tickers as gt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
from ta.utils import dropna


class Algo:
    def __init__(self):
       pass

    def execute(self):
        tickers = self.get_tickers()
        self.get_historical_data(tickers)
        self.get_technical_indicators()

    def get_tickers(self):
        # List of the stocks we are interested in analyzing. At the time of writing this, it narrows the list of
        # stocks down to 44. If you have a list of your own you would like to use just create a new list instead of
        # using this, for example: tickers = ["FB", "AMZN", ...]
        tickers = gt.get_tickers_filtered(mktcap_min=150000, mktcap_max=10000000)

        # Check that the amount of tickers isn't more than 2000
        print("The amount of stocks chosen to observe: " + str(len(tickers)))

        # These two lines remove the Stocks folder and then recreate it in order to remove old stocks. Make sure you
        # have created a Stocks Folder the first time you run this.
        shutil.rmtree("historical_data\\Bayesian_Logistic_Regression\\Stocks\\")
        os.mkdir("historical_data\\Bayesian_Logistic_Regression\\Stocks\\")
        return tickers

    def get_historical_data(self, tickers):
        # Holds the amount of API calls we executed
        num_api_calls = 0

        # This while loop is responsible for storing the historical data for each ticker in our list. Note that yahoo
        # finance sometimes incurs json.decode errors and because of this we are sleeping for 2 seconds after each
        # iteration, also if a call fails we are going to try to execute it again. Also, do not make more than 2,
        # 000 calls per hour or 48,000 calls per day or Yahoo Finance may block your IP. The clause "(
        # Amount_of_API_Calls < 1800)" below will stop the loop from making too many calls to the yfinance API.
        # Prepare for this loop to take some time. It is pausing for 2 seconds after importing each stock.

        # Used to make sure we don't waste too many API calls on one Stock ticker that could be having issues
        stock_failure = 0
        stocks_not_imported = 0

        # Used to iterate through our list of tickers
        i = 0
        while (i < len(tickers)) and (num_api_calls < 1800):
            try:
                stock = tickers[i]  # Gets the current stock ticker
                temp = yf.Ticker(str(stock))

                # Tells yfinance what kind of data we want about this stock (In this example, all of the historical
                # data)
                historical_data = temp.history(period="max")

                # Saves the historical data in csv format for further processing later
                historical_data.to_csv("historical_data\\Bayesian_Logistic_Regression\\Stocks\\" + stock + ".csv")

                # Pauses the loop for two seconds so we don't cause issues with Yahoo Finance's backend operations
                time.sleep(2)
                num_api_calls += 1
                stock_failure = 0
                i += 1  # Iteration to the next ticker
                print("Importing stock data:" + str(i))
            except ValueError:
                print("Yahoo Finance Backend Error, Attempting to Fix")
                if stock_failure > 5:  # Move on to the next ticker if the current ticker fails more than 5 times
                    i += 1
                    stocks_not_imported += 1
                num_api_calls += 1
                stock_failure += 1
        print("The amount of stocks we successfully imported: " + str(i - stocks_not_imported))

    def get_technical_indicators(self):
        # These two lines remove the Stocks folder and then recreate it in order to remove old stocks. Make sure you
        # have created a Stocks Folder the first time you run this.
        shutil.rmtree("historical_data\\Bayesian_Logistic_Regression\\Stocks_Sub\\")
        os.mkdir("historical_data\\Bayesian_Logistic_Regression\\Stocks_Sub\\")

        # Get the Y values
        # Creates a list of all csv filenames in the stocks folder
        list_files = (glob.glob("historical_data\\Bayesian_Logistic_Regression\\Stocks\\*.csv"))
        for interval in list_files:
            stock_name = ((os.path.basename(interval)).split(".csv")[0])
            data = pd.read_csv(interval)
            dropna(data)
            data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
            data = data.iloc[100:]
            close_prices = data['Close'].tolist()
            five_day_observation = []
            thirty_day_observation = []
            sixty_day_observation = []
            x = 0
            while x < (len(data)):
                if x < (len(data) - 5):
                    if ((close_prices[x + 1] + close_prices[x + 2] + close_prices[x + 3] + close_prices[x + 4] +
                         close_prices[x + 5]) / 5) > close_prices[x]:
                        five_day_observation.append(1)
                    else:
                        five_day_observation.append(0)
                else:
                    five_day_observation.append(0)
                x += 1
            y = 0
            while y < (len(data)):
                if y < (len(data) - 30):
                    thirty_day_calc = 0
                    y2 = 0
                    while y2 < 30:
                        thirty_day_calc = thirty_day_calc + close_prices[y + y2]
                        y2 += 1
                    if (thirty_day_calc / 30) > close_prices[y]:
                        thirty_day_observation.append(1)
                    else:
                        thirty_day_observation.append(0)
                else:
                    thirty_day_observation.append(0)
                y += 1
            z = 0
            while z < (len(data)):
                if z < (len(data) - 60):
                    sixty_day_calc = 0
                    z2 = 0
                    while z2 < 60:
                        sixty_day_calc = sixty_day_calc + close_prices[z + z2]
                        z2 += 1
                    if (sixty_day_calc / 60) > close_prices[z]:
                        sixty_day_observation.append(1)
                    else:
                        sixty_day_observation.append(0)
                else:
                    sixty_day_observation.append(0)
                z += 1
            data['Five_Day_Observation_Outcome'] = five_day_observation
            data['Thirty_Day_Observation_Outcome'] = thirty_day_observation
            data['Sixty_Day_Observation_Outcome'] = sixty_day_observation
            data.to_csv("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\" + stock_name + ".csv")
            print("Data for " + stock_name + " has been substantiated with technical features.")

# python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:01:13 2021

@author: hoctran
"""

#1. Import the libraries
import pandas as pd
import json
from datetime import datetime

#2. Parse the json
file = open('raw_10y.json')
fileData = json.load(file)

first = fileData['batchList'][0]

startTime = first['startTime']
interval = first['interval']
rates = first['rates'][1:]

#3. Mapping data
nRow = len(rates)
dates = []
for i in range(0, nRow):
    timestamp = startTime + interval * i
    dates.append(datetime.fromtimestamp(timestamp / 1000).strftime('%Y.%m.%d'))

dataframe = pd.DataFrame(list(zip(dates, rates)), columns =['date', 'rate'])

#4. Split to train and test set
trainSize = int(nRow * 0.8)
trainSet = dataframe.iloc[:trainSize, :]
testSet = dataframe.iloc[trainSize:, :]

#5. Write to files
trainSet.to_csv('rate_train.csv', index = False)
testSet.to_csv('rate_test.csv', index = False)

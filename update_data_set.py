# python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:01:13 2021

@author: hoctran
"""

import pandas as pd
import json
from datetime import datetime

file = open('raw_10y.json')
fileData = json.load(file)

first = fileData['batchList'][0]

startTime = first['startTime']
interval = first['interval']
rates = first['rates'][1:]


nRow = len(rates)
dates = []
for i in range(0, nRow):
    timestamp = startTime + interval * i
    dates.append(
        datetime.fromtimestamp(timestamp / 1000).strftime('%Y.%m.%d')
    )

dataframe = pd.DataFrame(list(zip(dates, rates)),
                         columns =['date', 'rate'])

trainSize = int(nRow * 0.8)
trainSet = dataframe.iloc[:nRow - trainSize, :]
testSet = dataframe.iloc[trainSize:, :]


trainSet.to_csv('rate_train.csv', index = False)
testSet.to_csv('rate_test.csv', index = False)


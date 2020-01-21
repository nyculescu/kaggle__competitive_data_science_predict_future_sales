# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('datasets'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# import os
# import csv

# ifile = os.path.abspath(os.path.join('input', 'sales_train.csv'))
# rows = []
# with open(ifile) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         rows.append(row)

# # use str() to avoid Exception has occurred: TypeError can only concatenate str (not "int") to str
# print("csv row length: " + str(readCSV.line_num)) 
# print("sizeof(row trasnformed into numpy obj): " + str(np.array(rows).nbytes) + " in bytes")


import os
import pandas as pd

def load_data():
    return pd.read_csv(os.path.abspath(os.path.join('input', 'sales_train.csv')))

sales_train = load_data()
print(sales_train)
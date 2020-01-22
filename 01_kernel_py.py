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
import csv
import numpy as np
import pandas as pd
from zlib import crc32
import matplotlib
matplotlib.use('agg') # explicitly asked for a non-GUI backend
import matplotlib.pyplot as plt

sales_train = pd.read_csv(os.path.abspath(os.path.join('input', 'sales_train.csv')))
sales_train.drop("date", axis=1, inplace=True)

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

sales_train_with_id = sales_train.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(sales_train_with_id, 0.2, "index")

train_set.plot(kind="scatter", x="date_block_num", y="item_cnt_day")
plt.show()
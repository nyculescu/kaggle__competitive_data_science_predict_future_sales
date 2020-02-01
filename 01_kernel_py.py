# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)


# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# Common imports
import numpy as np
import os
import csv
import pandas as pd


# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "kaggle__competitive_data_science_predict_future_sales"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# Print the entire number in Python from describe() function, but only for the numeric data
def describe_entire_nrs(datafr):
    desc = datafr.describe() # include='all' will not work
    desc.loc['count'] = desc.loc['count'].astype(int).astype(str)
    desc.iloc[1:] = desc.iloc[1:].applymap('{:.2f}'.format)
    return desc


# Used to generate the test set

from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

sales_train = pd.read_csv(os.path.abspath(os.path.join('input', 'sales_train.csv')))

# -------------------------------------------------------------------

sales_train_gb_mth = sales_train.drop(['item_id', 'item_price'], axis = 1).groupby(['shop_id', 'date_block_num']).sum().astype('int')
sales_train_gb_mth.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
sales_train_ = sales_train.copy()
sales_train_['item_cnt_month'] = int(0)
sales_train_.drop(["date"], inplace=True, axis=1)

# %%time
for row in sales_train_.iterrows():
  idx_s = int(sales_train_.loc[sales_train_.index[row[0]], "shop_id"])
  idx_m = int(sales_train_.loc[sales_train_.index[row[0]], "date_block_num"])

#   item_cnt_month = sales_train_gb_mth["item_cnt_month"][idx_s][idx_m] # this returns with the value 2017
  item_cnt_month = sales_train_gb_mth["item_cnt_month"]
  sales_train_.loc[sales_train_.index[row[0]], "item_cnt_month"] = int(item_cnt_month[idx_s][idx_m])

sales_train_.to_csv(r"input/sales_train_reshaped.csv")
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
from pandas.plotting import scatter_matrix
import datetime
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

# To plot pretty figures
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

colab_path = "drive/My Drive/Colab Notebooks/kaggle_competitive_data_science_predict_future_sales"

# -------------------------------------------------------------------

sales_train_ = pd.read_csv(os.path.abspath(os.path.join('input', 'sales_train_reshaped.csv')))

sales_train_.drop('Unnamed: 0', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(sales_train_, train_size=0.8, shuffle=False) 

sales_train_ts = train_set.copy()

sales_train_ts_lb = sales_train_ts['item_cnt_month'].copy()
sales_train_ts.drop('item_cnt_month', axis=1, inplace=True)

from sklearn.preprocessing import OneHotEncoder
item_cat_1hot = OneHotEncoder().fit_transform(sales_train_ts[["item_id"]])
shop_cat_1hot = OneHotEncoder().fit_transform(sales_train_ts[["shop_id"]])
attr_to_be_dropped = {
    "item_id",
    "shop_id"
    }
sales_train_num = sales_train_ts.drop(attr_to_be_dropped, axis=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline([ # a small pipeline for the numerical attributes
        ('imputer', SimpleImputer(strategy="median")),
        # ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
sales_train_num_tr = num_pipeline.fit_transform(sales_train_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(sales_train_num)
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("itm", OneHotEncoder(), ["item_id"]),
        ("shp", OneHotEncoder(), ["shop_id"]),
    ])
sales_train_ts_prep = full_pipeline.fit_transform(sales_train_ts)

from sklearn.ensemble import RandomForestRegressor
print(datetime.datetime.now())
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(sales_train_ts_prep, sales_train_ts_lb)
print(datetime.datetime.now())

sales_train_predictions = forest_reg.predict(sales_train_ts_prep)
forest_mse = mean_squared_error(sales_train_ts_lb, sales_train_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

joblib.dump(forest_reg, os.path.abspath(os.path.join('models', 'forest_reg.pkl')))
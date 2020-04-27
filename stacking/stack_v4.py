import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# from tqdm import tqdm_notebook

shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
train = pd.read_csv('sales_train.csv')
items = pd.read_csv('items.csv')
cats = pd.read_csv('item_categories.csv')

# Duplicate shops correction

train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median
# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
train.loc[train.shop_id == 40, "shop_id"] = 39
test.loc[test.shop_id == 40, "shop_id"] = 39

# we don't do label encoding here, since we will use catboost

# shops

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск' 

category = []
for cat in shops.category.unique():
    if len(shops[shops.category == cat]) > 4:
        category.append(cat)
        
shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" )
shops = shops[["shop_id", "category", "city"]]

# item categories

cat_list = list(cats.item_category_name)

for i in range(1,8):
    cat_list[i] = 'Access'

for i in range(10,18):
    cat_list[i] = 'Consoles'

for i in range(18,25):
    cat_list[i] = 'Consoles Games'

for i in range(26,28):
    cat_list[i] = 'phone games'

for i in range(28,32):
    cat_list[i] = 'CD games'

for i in range(32,37):
    cat_list[i] = 'Card'

for i in range(37,43):
    cat_list[i] = 'Movie'

for i in range(43,55):
    cat_list[i] = 'Books'

for i in range(55,61):
    cat_list[i] = 'Music'

for i in range(61,73):
    cat_list[i] = 'Gifts'

for i in range(73,79):
    cat_list[i] = 'Soft'

cats["subtype"] = cat_list

cats = cats[["item_category_id", "subtype"]]

import re
def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x

items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items = items.fillna('0')

items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")
items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
items.loc[ items.type == "", "type"] = "mac"
items.type = items.type.apply( lambda x: x.replace(" ", "") )
items.loc[ (items.type == 'pc' ) | (items.type == 'pс') |  (items.type == 'рс') | (items.type == "pc"), "type" ] = "pc"
items.loc[ items.type == 'рs3' , "type"] = "ps3"

drop_cols = []
group_sum = items.groupby(["type"]).agg({"item_id": "count"})
group_sum = group_sum.reset_index()
for cat in group_sum.type.unique():
#     print(group_sum.loc[(group_sum.type == cat), "item_id"].values[0])
    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
        drop_cols.append(cat)
items.type = items.type.apply( lambda x: "etc" if (x in drop_cols) else x )
items.drop(["item_name", "name1", "name2"],axis = 1, inplace= True)

from itertools import product
import time
ts = time.time()
matrix = []
cols  = ["date_block_num", "shop_id", "item_id"]
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )

matrix = pd.DataFrame( np.vstack(matrix), columns = cols )
matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
matrix["item_id"] = matrix["item_id"].astype(np.int16)
matrix.sort_values( cols, inplace = True )
time.time()- ts

train["revenue"] = train["item_cnt_day"] * train["item_price"]
ts = time.time()
group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
group.columns = ["item_cnt_month"]
group.reset_index( inplace = True)
matrix = pd.merge( matrix, group, on = cols, how = "left" )
matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).clip(0,20).astype(np.float16) #do we clip here??
test["date_block_num"] = 34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["shop_id"] = test.shop_id.astype(np.int8)
test["item_id"] = test.item_id.astype(np.int16)
matrix = pd.concat([matrix, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
matrix.fillna( 0, inplace = True )
matrix = pd.merge( matrix, shops, on = ["shop_id"], how = "left" )
matrix = pd.merge(matrix, items, on = ["item_id"], how = "left")
matrix = pd.merge( matrix, cats, on = ["item_category_id"], how = "left" )
time.time() - ts

def lag_feature( df,lags, cols ):
    for col in cols:
        print(col)
        tmp = df[["date_block_num", "shop_id","item_id",col ]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

# %% [code]
matrix

# %% [code]
ts = time.time()


# single feature mean encodings, e.g. groupby only item, will be automatically 
# done in catboost. Note: catboost will not create lagged features since 
# it will take ALL rows into account
# lags only make sense if we partition also by date, so we are doing mean
# encodings for features / feature combinations combined with date, this needs
# to be specified separately, see next point

# catboost cannot mean encode date_block, since we partition the set by 
# exactly this variable, so category 35 would never be seen in training

group = matrix.groupby( ["date_block_num"] ).agg({"item_cnt_month" : ["mean"]})
group.columns = ["date_avg_item_cnt"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num"], how = "left")

del group
gc.collect()

matrix.date_avg_item_cnt = matrix["date_avg_item_cnt"].astype(np.float16)
matrix = lag_feature( matrix, [1,2,3], ["date_avg_item_cnt"] )
matrix.drop( ["date_avg_item_cnt"], axis = 1, inplace = True )

# date and shop and item

matrix = lag_feature( matrix, [1,2,3], ["item_cnt_month"] )

# date and shop

group = matrix.groupby( ["date_block_num","shop_id"] ).agg({"item_cnt_month" : ["mean"]})
group.columns = ["avg_sell_month_per_shop"]
group.reset_index(inplace = True)
matrix = pd.merge(matrix, group, on = ["date_block_num","shop_id"], how = "left")
del group
gc.collect()
matrix.avg_sell_month_per_shop = matrix["avg_sell_month_per_shop"].astype(np.float16)
matrix = lag_feature( matrix, [1,2,3], ["avg_sell_month_per_shop"] )
matrix.drop( ["avg_sell_month_per_shop"], axis = 1, inplace = True )

# date and item

group = matrix.groupby( ["date_block_num","item_id"] ).agg({"item_cnt_month" : ["mean"]})
group.columns = ["avg_sell_month_per_item"]
group.reset_index(inplace = True)
matrix = pd.merge(matrix, group, on = ["date_block_num","item_id"], how = "left")
del group
gc.collect()
matrix.avg_sell_month_per_item = matrix["avg_sell_month_per_item"].astype(np.float16)
matrix = lag_feature( matrix, [1,2,3], ["avg_sell_month_per_item"] )
matrix.drop( ["avg_sell_month_per_item"], axis = 1, inplace = True )

# date and subtype

group = matrix.groupby( ["date_block_num","subtype"] ).agg({"item_cnt_month" : ["mean"]})
group.columns = ["avg_sell_month_per_subtype"]
group.reset_index(inplace = True)
matrix = pd.merge(matrix, group, on = ["date_block_num","subtype"], how = "left")
del group
gc.collect()
matrix.avg_sell_month_per_subtype = matrix["avg_sell_month_per_subtype"].astype(np.float16)
matrix = lag_feature( matrix, [1,2,3], ["avg_sell_month_per_subtype"] )
matrix.drop( ["avg_sell_month_per_subtype"], axis = 1, inplace = True )

# date and shop and subtype

group = matrix.groupby(['date_block_num', 'shop_id', 'subtype']).agg({'item_cnt_month': ['mean']})
group.columns = ['avg_sell_month_per_shop_subtype']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype'], how='left')
del group
gc.collect()
matrix.avg_sell_month_per_shop_subtype = matrix['avg_sell_month_per_shop_subtype'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3], ['avg_sell_month_per_shop_subtype'])
matrix.drop(['avg_sell_month_per_shop_subtype'], axis=1, inplace=True)

# date and city

group = matrix.groupby(['date_block_num', 'city']).agg({'item_cnt_month': ['mean']})
group.columns = ['avg_sell_month_per_city']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', "city"], how='left')
del group
gc.collect()
matrix.avg_sell_month_per_city = matrix['avg_sell_month_per_city'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3], ['avg_sell_month_per_city'])
matrix.drop(['avg_sell_month_per_city'], axis=1, inplace=True)

# date and city and item

group = matrix.groupby(['date_block_num', 'item_id', 'city']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'avg_sell_month_per_city_item' ]
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city'], how='left')
del group
gc.collect()
matrix.avg_sell_month_per_city_item = matrix['avg_sell_month_per_city_item'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3], ['avg_sell_month_per_city_item'])
matrix.drop(['avg_sell_month_per_city_item'], axis=1, inplace=True)

# date and city and subtype

group = matrix.groupby(['date_block_num', 'subtype', 'city']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'avg_sell_month_per_city_subtype' ]
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype', 'city'], how='left')
del group
gc.collect()
matrix.avg_sell_month_per_city_subtype = matrix['avg_sell_month_per_city_subtype'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3], ['avg_sell_month_per_city_subtype'])
matrix.drop(['avg_sell_month_per_city_subtype'], axis=1, inplace=True)

time.time() - ts

ts = time.time()

## creating delta price lag

group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})
group.columns = ["item_avg_item_price"]
group.reset_index(inplace = True)
matrix = matrix.merge( group, on = ["item_id"], how = "left" )
matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)
group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )
group.columns = ["date_item_avg_item_price"]
group.reset_index(inplace = True)
matrix = matrix.merge(group, on = ["date_block_num","item_id"], how = "left")
del group
gc.collect()
matrix["date_item_avg_item_price"] = matrix.date_item_avg_item_price.astype(np.float16)
lags = [1,2,3]
matrix = lag_feature( matrix, lags, ["date_item_avg_item_price"] )
for i in lags:
    matrix["delta_price_lag_" + str(i) ] = (matrix["date_item_avg_item_price_lag_" + str(i)]- matrix["item_avg_item_price"] )/ matrix["item_avg_item_price"]

# selects the first existing one; some products may not have been sold in previous month
    
def select_trends(row) :
    for i in lags:
        if row["delta_price_lag_" + str(i)]:
            return row["delta_price_lag_" + str(i)]
    return 0

# last month when it was available, how did the average item price compare to the grand mean price of the same item

matrix["delta_price_lag"] = matrix.apply(select_trends, axis = 1)
matrix["delta_price_lag"] = matrix.delta_price_lag.astype( np.float16 )
matrix["delta_price_lag"].fillna( 0 ,inplace = True)
features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]
for i in lags:
    features_to_drop.append("date_item_avg_item_price_lag_" + str(i) )
    features_to_drop.append("delta_price_lag_" + str(i) )
matrix.drop(features_to_drop, axis = 1, inplace = True)

## creating delta revenue

group = train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })
group.columns = ["date_shop_revenue"]
group.reset_index(inplace = True)
matrix = matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" )
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

# mistake in template script, aggregation was w.r.t date_block_num and not date_shop_revenue

group = group.groupby(["shop_id"]).agg({ "date_shop_revenue":["mean"] })
group.columns = ["shop_avg_revenue"]
group.reset_index(inplace = True )
matrix = matrix.merge( group, on = ["shop_id"], how = "left" )
del group
gc.collect()
matrix["shop_avg_revenue"] = matrix.shop_avg_revenue.astype(np.float32)
matrix["delta_revenue"] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix["delta_revenue"] = matrix["delta_revenue"]. astype(np.float16)

# how was last month's revenue compared to the grand mean (across time) revenue of the same shop

matrix = lag_feature(matrix, [1], ["delta_revenue"])

# adding a new feature

matrix = lag_feature(matrix, [1], ["date_shop_revenue"])

# notice how we are not dropping the lag, but drop everything else

matrix["delta_revenue_lag_1"] = matrix["delta_revenue_lag_1"].astype(np.float16)
matrix.drop( ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"] ,axis = 1, inplace = True)

## creating delta transactions

group = train.groupby( ["date_block_num","shop_id"] ).agg({"item_cnt_day": ["sum"] }) #month
group.columns = ["date_shop_transactions"]
group.reset_index(inplace = True)
matrix = matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" )
matrix['date_shop_transactions'] = matrix['date_shop_transactions'].astype(np.float32)
group = group.groupby(["shop_id"]).agg({ "date_shop_transactions":["mean"] })
group.columns = ["shop_avg_transactions"]
group.reset_index(inplace = True )
matrix = matrix.merge( group, on = ["shop_id"], how = "left" )
del group
gc.collect()
matrix["shop_avg_transactions"] = matrix.shop_avg_transactions.astype(np.float32)
matrix["delta_transactions"] = (matrix['date_shop_transactions'] - matrix['shop_avg_transactions']) / matrix['shop_avg_transactions']
matrix["delta_transactions"] = matrix["delta_transactions"]. astype(np.float16)

# how was last month's transactions compared to the grand mean (across time) transactions of the same shop
# we can drop date_shop_transactions, already calculated with mean and lagged

matrix = lag_feature(matrix, [1], ["delta_transactions"])
matrix["delta_transactions_lag_1"] = matrix["delta_transactions_lag_1"].astype(np.float16)
matrix.drop( ["date_shop_transactions", "shop_avg_transactions", "delta_transactions"] ,axis = 1, inplace = True)

## creating delta_transactions_item

group = train.groupby( ["date_block_num","item_id"] ).agg({"item_cnt_day": ["sum"] }) #month
group.columns = ["date_item_transactions"]
group.reset_index(inplace = True)
matrix = matrix.merge( group , on = ["date_block_num", "item_id"], how = "left" )
matrix['date_item_transactions'] = matrix['date_item_transactions'].astype(np.float32)
group = group.groupby(["item_id"]).agg({ "date_item_transactions":["mean"] })
group.columns = ["item_avg_transactions"]
group.reset_index(inplace = True )
matrix = matrix.merge( group, on = ["item_id"], how = "left" )
del group
gc.collect()
matrix["item_avg_transactions"] = matrix.item_avg_transactions.astype(np.float32)
matrix["delta_transactions_item"] = (matrix['date_item_transactions'] - matrix['item_avg_transactions']) / matrix['item_avg_transactions']
matrix["delta_transactions_item"] = matrix["delta_transactions_item"]. astype(np.float16)

# how was last month's transactions compared to the grand mean (across time) transactions of the same item
# we can drop date_item_transactions, already calculated with mean and lagged

matrix = lag_feature(matrix, [1], ["delta_transactions_item"])
matrix["delta_transactions_item_lag_1"] = matrix["delta_transactions_item_lag_1"].astype(np.float16)
matrix.drop( ["date_item_transactions", "item_avg_transactions", "delta_transactions_item"] ,axis = 1, inplace = True)
time.time() - ts

# we technically could leave NaNs in the matrix, as catboost handles them

matrix.fillna( 0, inplace = True )

# Holiday and stock exchange features

rts_max = {
    0: 1635, 1: 1628, 2: 1542, 
    3: 1453, 4: 1461, 5: 1322,
    6: 1393, 7: 1356, 8: 1478, 
    9: 1518, 10: 1475, 11: 1454,
    12: 1401, 13: 1353, 14: 1226,
    15: 1235, 16: 1335, 17: 1421,
    18: 1403, 19: 1275, 20: 1257,
    21: 1120, 22: 1078, 23: 958,
    24: 821, 25: 929, 26: 917,
    27: 1061, 28: 1082, 29: 981,
    30: 932, 31: 848, 32: 837,
    33: 885, 34: 897
}

matrix['rts_max'] = matrix.date_block_num.map(rts_max)

# weekends = {
#     0: 8, 1: 8, 2: 10, 
#     3: 8, 4: 8, 5: 10, 
#     6: 8, 7: 9, 8: 9,
#     9: 8, 10: 9, 11:9, 
#     12: 8, 13: 8, 14: 10,
#     15: 8, 16: 9, 17: 9,
#     18: 8, 19: 10, 20: 8,
#     21: 8, 22: 10, 23: 8,
#     24: 9, 25: 8, 26: 9,
#     27: 8, 28: 10, 29: 8,
#     30: 8, 31: 10, 32: 8,
#     33: 9, 34: 9
# }

# matrix['weekends'] = matrix.date_block_num.map(weekends)

# days = {
    # 0: 31, 1: 28, 2: 31, 
    # 3: 30, 4: 31, 5: 30,
    # 6: 31, 7: 31, 8: 30,
    # 9: 31, 10: 30, 11: 31,
    # 12: 31, 13: 28, 14: 31,
    # 15: 30, 16: 31, 17: 30,
    # 18: 31, 19: 31, 20: 30,
    # 21: 31, 22: 30, 23: 31,
    # 24: 31, 25: 28, 26: 31,
    # 27: 30, 28: 31, 29: 30,
    # 30: 31, 31: 31, 32: 30,
    # 33: 31, 34: 30
# }

# matrix['days'] = matrix.date_block_num.map(days)

# # holidays excluding the weekends

# holidays = {
#     0: 6, 1: 0, 2: 1, 
#     3: 0, 4: 5, 5: 1,
#     6: 0, 7: 0, 8: 0, 
#     9: 0, 10: 1, 11: 0,
#     12: 7, 13: 0, 14: 1,
#     15: 0, 16: 2, 17: 1,
#     18: 0, 19: 0, 20: 0,
#     21: 0, 22: 1, 23: 0,
#     24: 7, 25: 1, 26: 1,
#     27: 0, 28: 2, 29: 1,
#     30: 0, 31: 0, 32: 0,
#     33: 0, 34: 1
# }

# matrix['holidays'] = matrix.date_block_num.map(holidays)

matrix["month"] = matrix["date_block_num"] % 12

# these features are redundant, as can be guessed from zeros in other features, but 
# adding these to make the boosting algorithm's life easier

matrix["item_shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id","shop_id"])["date_block_num"].transform('min')
matrix["item_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id"])["date_block_num"].transform('min')
matrix["shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(["shop_id"])["date_block_num"].transform('min')

# %% [code]
matrix["rts_max"] = matrix["rts_max"].astype(np.int16)
# matrix["weekends"] = matrix["weekends"].astype(np.int8)
# matrix["days"] = matrix["days"].astype(np.int8)
# matrix["holidays"] = matrix["holidays"].astype(np.int8)

pd.set_option('use_inf_as_na', True)
matrix = matrix.fillna(0)

matrix = matrix[matrix["date_block_num"] > 18] #starting from 19 only now
matrix

# %% [code]
from sklearn.preprocessing import LabelEncoder
matrix["category"] = LabelEncoder().fit_transform( matrix.category )
matrix["city"] = LabelEncoder().fit_transform( matrix.city )
matrix["name3"] = LabelEncoder().fit_transform( matrix.name3 )
matrix["type"] = LabelEncoder().fit_transform( matrix.type )
matrix["subtype"] = LabelEncoder().fit_transform( matrix.subtype )

# %% [code]
from sklearn.model_selection import KFold

X_train = matrix.loc[(matrix['date_block_num'] < 34) & (matrix['date_block_num'] > 30)].drop(['item_cnt_month'], axis=1) #date_block_num is not useful
Y_train = matrix.loc[(matrix['date_block_num'] < 34) & (matrix['date_block_num'] > 30), 'item_cnt_month']
X_test = matrix.loc[matrix['date_block_num'] == 34].drop(['item_cnt_month'], axis=1)

# dates_train = matrix['date_block_num']

# %% [code]
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

ts = time.time()

model = XGBRegressor(
    max_depth=8,
    n_estimators=50,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
#     tree_method='gpu_hist',
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train)], 
    verbose=True)

time.time() - ts

pred_gb = model.predict(X_test).clip(0,20)

# %% [code]
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))

# %% [code]
X_train.drop('date_block_num', axis=1, inplace=True)
X_test.drop('date_block_num', axis=1, inplace=True)

cat_features = ['shop_id', 'item_id', 'category', 'city', 'item_category_id', 'name3', 'type', 'subtype', 'month', 'item_shop_first_sale', 'item_first_sale', 'shop_first_sale']
num_features = [item for item in X_train.columns if item not in cat_features]

X_train[cat_features] = X_train[cat_features].astype(object)
X_train[num_features] = X_train[num_features].astype(np.float32)
Y_train = Y_train.astype(np.float32)
X_test[cat_features] = X_test[cat_features].astype(object)
X_test[num_features] = X_test[num_features].astype(np.float32)

# %% [code]
from catboost import CatBoostRegressor

cb_model = CatBoostRegressor(
    iterations = 100,
    bootstrap_type='Bernoulli',
    max_ctr_complexity=2,
    eval_metric ='RMSE',
    rsm = 0.8,
    subsample = 0.8,
    max_depth = 4,
    cat_features = cat_features
)
cb_model.fit(
    X_train, Y_train,
    verbose = True
)

# %% [code]
pred_cb = cb_model.predict(X_test).clip(0,20)

importances = cb_model.get_feature_importance(prettified=True)
print(importances)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=15, max_depth=10, n_jobs=-1)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test.values).clip(0,20) 

pred_train = rf.predict(X_train.values)
rmse = np.sqrt(mean_squared_error(pred_train.clip(0,20), Y_train))
print('Train RMSE for random forest is %f' % rmse)

# %% [code]
gc.collect()

# %% [code]
X_test_level2 = np.c_[pred_gb, pred_cb, pred_rf]

# That is how we get target for the 2nd level dataset
Y_train_level2 = matrix.loc[matrix['date_block_num'].isin([22, 31, 32, 33]), 'item_cnt_month']

# dates_train_level2 = dates_train[dates_train.isin([22, 33])]

# And here we create 2nd level feeature matrix, init it with zeros first
# X_train_level2 = np.zeros([Y_train_level2.shape[0], 3])

# Now fill `X_train_level2` with metafeatures

iter = 0
for cur_block_num in [22, 33]:
    
    print(cur_block_num)
    
    '''
        1. Split `X_train` into parts
           Remember, that corresponding dates are stored in `dates_train` 
        2. Fit linear regression 
        3. Fit LightGBM and put predictions          
        4. Store predictions from 2. and 3. in the right place of `X_train_level2`. 
           You can use `dates_train_level2` for it
           Make sure the order of the meta-features is the same as in `X_test_level2`
    '''      
    
    #  YOUR CODE GOES HERE
    
    matrix["category"] = LabelEncoder().fit_transform( matrix.category )
    matrix["city"] = LabelEncoder().fit_transform( matrix.city )
    matrix["name3"] = LabelEncoder().fit_transform( matrix.name3 )
    matrix["type"] = LabelEncoder().fit_transform( matrix.type )
    matrix["subtype"] = LabelEncoder().fit_transform( matrix.subtype )
    
    X_train = matrix.loc[(matrix['date_block_num'] < cur_block_num) & (matrix['date_block_num'] > (cur_block_num-4))].drop(['item_cnt_month'], axis=1)
    Y_train = matrix.loc[(matrix['date_block_num'] < cur_block_num) & (matrix['date_block_num'] > (cur_block_num-4)), 'item_cnt_month']
    X_test = matrix.loc[matrix['date_block_num'] == cur_block_num].drop(['item_cnt_month'], axis=1)
    
    #xgb
    
    model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train)], 
    verbose=True)
    pred_gb = model.predict(X_test).clip(0,20)
    
    #cb
    
    X_train.drop('date_block_num', axis=1, inplace=True)
    X_test.drop('date_block_num', axis=1, inplace=True)

    cat_features = ['shop_id', 'item_id', 'category', 'city', 'item_category_id', 'name3', 'type', 'subtype', 'month', 'item_shop_first_sale', 'item_first_sale', 'shop_first_sale']
    num_features = [item for item in X_train.columns if item not in cat_features]

    X_train[cat_features] = X_train[cat_features].astype(object)
    X_train[num_features] = X_train[num_features].astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test[cat_features] = X_test[cat_features].astype(object)
    X_test[num_features] = X_test[num_features].astype(np.float32)
    
    cb_model.fit(
    X_train, Y_train,
    verbose = True)
    pred_cb = cb_model.predict(X_test).clip(0,20)
	
    rf.fit(X_train, Y_train)
    pred_rf = rf.predict(X_test.values).clip(0,20) 
	
    if iter==0:
	    X_train_level2 = np.c_[pred_gb, pred_cb, pred_rf]
    else:
	    X_train_level2 = np.vstack((X_train_level2,np.c_[pred_gb, pred_cb, pred_rf]))
		
    iter = iter + 1
    gc.collect()

# %% [code]
meta_model = LinearRegression()
meta_model.fit(X_train_level2, Y_train_level2)

train_preds = meta_model.predict(X_train_level2) # YOUR CODE GOES HERE
rmse = np.sqrt(mean_squared_error(train_preds, Y_train_level2))
print('Train RMSE for linreg is %f' % rmse)

Y_test = meta_model.predict(X_test_level2) # YOUR CODE GOES HERE

# %% [code]
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('short_stack_submission.csv', index=False)

print('model weights',
          'xgb',
          '{:.1%}'.format(meta_model.coef_[0]),
          'cb',
          '{:.1%}'.format(meta_model.coef_[1]),
          'rf',
          '{:.1%}'.format(meta_model.coef_[2]))


# save predictions for an ensemble
# pickle.dump(Y_test, open('catboost_test.pickle', 'wb'))
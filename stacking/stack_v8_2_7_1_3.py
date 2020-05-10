#scores 0.88778

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



print(train.shape[0])
print(len( train[train.item_cnt_day >999 ] ))
print( len(train[ train.item_cnt_day > 500  ]) )
print(len(train[train.item_price >100000 ]))
train = train[(train.item_price < 100000 )& (train.item_cnt_day < 1000)]
print(train.shape[0])



# %% [code]
train = train[train.item_price > 0].reset_index(drop = True)
train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0


# %% [code]
shops

# %% [code]
train.loc[train.shop_id == 0, "shop_id"] = 57
test.loc[test.shop_id == 0 , "shop_id"] = 57
train.loc[train.shop_id == 1, "shop_id"] = 58
test.loc[test.shop_id == 1 , "shop_id"] = 58
train.loc[train.shop_id == 11, "shop_id"] = 10
test.loc[test.shop_id == 11, "shop_id"] = 10
train.loc[train.shop_id == 40, "shop_id"] = 39
test.loc[test.shop_id == 40, "shop_id"] = 39

# %% [code]
shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
shops.loc[shops.city == "!Якутск", "city"] = "Якутск"


# %% [code]
category = []
for cat in shops.category.unique():
    print(cat, len(shops[shops.category == cat]) )
    if len(shops[shops.category == cat]) > 4:
        category.append(cat)

# %% [code]
shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" )

# %% [code]
for cat in shops.category.unique():
    print(cat, len(shops[shops.category == cat]) )

# %% [code]
from sklearn.preprocessing import LabelEncoder
shops["shop_category"] = LabelEncoder().fit_transform( shops.category )
shops["shop_city"] = LabelEncoder().fit_transform( shops.city )

# %% [code]
shops = shops[["shop_id", "shop_category", "shop_city"]]

# %% [code]
cats["type_code"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)
cats.loc[ (cats.type_code == "Игровые")| (cats.type_code == "Аксессуары"), "category" ] = "Игры"

# %% [code]
cats.shape

# %% [code]
category = []
for cat in cats.type_code.unique():
    print(cat, len(cats[cats.type_code == cat]))
    if len(cats[cats.type_code == cat]) > 4: 
        category.append( cat )

# %% [code]
cats.type_code = cats.type_code.apply(lambda x: x if (x in category) else "etc")

# %% [code]
for cat in cats.type_code.unique():
    print(cat, len(cats[cats.type_code == cat]))

# %% [code]
cats.type_code = LabelEncoder().fit_transform(cats.type_code)
cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))
cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats["subtype_code"] = LabelEncoder().fit_transform( cats["subtype"] )
cats = cats[["item_category_id", "subtype_code", "type_code"]]

# %% [code]
cats.head()

# %% [code]
import re
def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x

# %% [code]

items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items = items.fillna('0')

items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))
items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")

# %% [code]
items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
items.loc[ items.type == "", "type"] = "mac"
items.type = items.type.apply( lambda x: x.replace(" ", "") )
items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
items.loc[ items.type == 'рs3' , "type"] = "ps3"


# %% [code]
group_sum = items.groupby(["type"]).agg({"item_id": "count"})
print( group_sum.reset_index() )
group_sum = group_sum.reset_index()

# %% [code]
drop_cols = []
for cat in group_sum.type.unique():
#     print(group_sum.loc[(group_sum.type == cat), "item_id"].values[0])
    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
        drop_cols.append(cat)

# %% [code]
drop_cols

# %% [code]
items.head()

# %% [code]
items.name2 = items.name2.apply( lambda x: "etc" if (x in drop_cols) else x )
items = items.drop(["type"], axis = 1)

# %% [code]
items.name2 = LabelEncoder().fit_transform(items.name2)
items.name3 = LabelEncoder().fit_transform(items.name3)

items.drop(["item_name", "name1"],axis = 1, inplace= True)
items.head()

# %% [code]
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

# %% [code]
train["revenue"] = train["item_cnt_day"] * train["item_price"]

# %% [code]
ts = time.time()
group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
group.columns = ["item_cnt_month"]
group.reset_index( inplace = True)
matrix = pd.merge( matrix, group, on = cols, how = "left" )
matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).clip(0,20).astype(np.float16)
time.time() - ts


# %% [code]
test["date_block_num"] = 34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["shop_id"] = test.shop_id.astype(np.int8)
test["item_id"] = test.item_id.astype(np.int16)


# %% [code]
ts = time.time()

matrix = pd.concat([matrix, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
matrix.fillna( 0, inplace = True )
time.time() - ts

# %% [code]


# %% [code]


# %% [code]
ts = time.time()
matrix = pd.merge( matrix, shops, on = ["shop_id"], how = "left" )
matrix = pd.merge(matrix, items, on = ["item_id"], how = "left")
matrix = pd.merge( matrix, cats, on = ["item_category_id"], how = "left" )
matrix["shop_city"] = matrix["shop_city"].astype(np.int8)
matrix["shop_category"] = matrix["shop_category"].astype(np.int8)
matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)
matrix["subtype_code"] = matrix["subtype_code"].astype(np.int8)
matrix["name2"] = matrix["name2"].astype(np.int8)
matrix["name3"] = matrix["name3"].astype(np.int16)
matrix["type_code"] = matrix["type_code"].astype(np.int8)
time.time() - ts

# %% [code]
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
ts = time.time()

matrix = lag_feature( matrix, [1,2,3], ["item_cnt_month"] )
time.time() - ts

# %% [code]
ts = time.time()
group = matrix.groupby( ["date_block_num"] ).agg({"item_cnt_month" : ["mean"]})
group.columns = ["date_avg_item_cnt"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num"], how = "left")
matrix.date_avg_item_cnt = matrix["date_avg_item_cnt"].astype(np.float16)
matrix = lag_feature( matrix, [1], ["date_avg_item_cnt"] )
matrix.drop( ["date_avg_item_cnt"], axis = 1, inplace = True )
time.time() - ts

# %% [code]
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix.date_item_avg_item_cnt = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3], ['date_item_avg_item_cnt'])
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

# %% [code]


# %% [code]
ts = time.time()
group = matrix.groupby( ["date_block_num","shop_id"] ).agg({"item_cnt_month" : ["mean"]})
group.columns = ["date_shop_avg_item_cnt"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num","shop_id"], how = "left")
matrix.date_avg_item_cnt = matrix["date_shop_avg_item_cnt"].astype(np.float16)
matrix = lag_feature( matrix, [1,2,3], ["date_shop_avg_item_cnt"] )
matrix.drop( ["date_shop_avg_item_cnt"], axis = 1, inplace = True )
time.time() - ts

# %% [code]


# %% [code]
ts = time.time()
group = matrix.groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})
group.columns = ["date_shop_item_avg_item_cnt"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num","shop_id","item_id"], how = "left")
matrix.date_avg_item_cnt = matrix["date_shop_item_avg_item_cnt"].astype(np.float16)
matrix = lag_feature( matrix, [1,2,3], ["date_shop_item_avg_item_cnt"] )
matrix.drop( ["date_shop_item_avg_item_cnt"], axis = 1, inplace = True )
time.time() - ts

# %% [code]
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix.date_shop_subtype_avg_item_cnt = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], ['date_shop_subtype_avg_item_cnt'])
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

# %% [code]
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_city']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_city_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', "shop_city"], how='left')
matrix.date_city_avg_item_cnt = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], ['date_city_avg_item_cnt'])
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

# %% [code]
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id', 'shop_city']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'shop_city'], how='left')
matrix.date_item_city_avg_item_cnt = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], ['date_item_city_avg_item_cnt'])
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

# %% [code]
ts = time.time()
group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})
group.columns = ["item_avg_item_price"]
group.reset_index(inplace = True)

matrix = matrix.merge( group, on = ["item_id"], how = "left" )
matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)


group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )
group.columns = ["date_item_avg_item_price"]
group.reset_index(inplace = True)

matrix = matrix.merge(group, on = ["date_block_num","item_id"], how = "left")
matrix["date_item_avg_item_price"] = matrix.date_item_avg_item_price.astype(np.float16)
lags = [1, 2, 3]
matrix = lag_feature( matrix, lags, ["date_item_avg_item_price"] )
for i in lags:
    matrix["delta_price_lag_" + str(i) ] = (matrix["date_item_avg_item_price_lag_" + str(i)]- matrix["item_avg_item_price"] )/ matrix["item_avg_item_price"]

def select_trends(row) :
    for i in lags:
        if row["delta_price_lag_" + str(i)]:
            return row["delta_price_lag_" + str(i)]
    return 0

matrix["delta_price_lag"] = matrix.apply(select_trends, axis = 1)
matrix["delta_price_lag"] = matrix.delta_price_lag.astype( np.float16 )
matrix["delta_price_lag"].fillna( 0 ,inplace = True)

features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]
for i in lags:
    features_to_drop.append("date_item_avg_item_price_lag_" + str(i) )
    features_to_drop.append("delta_price_lag_" + str(i) )
matrix.drop(features_to_drop, axis = 1, inplace = True)
time.time() - ts


# %% [code]
ts = time.time()
group = train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })
group.columns = ["date_shop_revenue"]
group.reset_index(inplace = True)

matrix = matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" )
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(["shop_id"]).agg({ "date_shop_revenue":["mean"] })
group.columns = ["shop_avg_revenue"]
group.reset_index(inplace = True )

matrix = matrix.merge( group, on = ["shop_id"], how = "left" )
matrix["shop_avg_revenue"] = matrix.shop_avg_revenue.astype(np.float32)
matrix["delta_revenue"] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix["delta_revenue"] = matrix["delta_revenue"]. astype(np.float32)

matrix = lag_feature(matrix, [1], ["delta_revenue"])
matrix["delta_revenue_lag_1"] = matrix["delta_revenue_lag_1"].astype(np.float32)
matrix.drop( ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"] ,axis = 1, inplace = True)
time.time() - ts



# %% [code]
matrix

# %% [code]
matrix.head().T

# %% [code]
matrix["month"] = matrix["date_block_num"] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

# %% [code]
matrix["days"] = matrix["month"].map(days).astype(np.int8)

# %% [code]
ts = time.time()
matrix["item_shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id","shop_id"])["date_block_num"].transform('min')
matrix["item_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id"])["date_block_num"].transform('min')
time.time() - ts


pd.set_option('use_inf_as_na', True)
matrix = matrix.fillna(0)

matrix = matrix[matrix["date_block_num"] > 3] 

# %% [code]
from sklearn.model_selection import KFold

X_train = matrix.loc[(matrix['date_block_num'] < 34) & (matrix['date_block_num'] > 15)].drop(['item_cnt_month'], axis=1) #date_block_num is not useful
Y_train = matrix.loc[(matrix['date_block_num'] < 34) & (matrix['date_block_num'] > 15), 'item_cnt_month']
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
    max_depth=7,
    n_estimators=100,
    min_child_weight=6, 
	gamma = 0.05,
    #colsample_bytree=0.8, 
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

cat_features = ['shop_id', 'item_id', 'shop_category', 'shop_city', 'item_category_id', 'name2', 'name3', 'type_code', 'subtype_code', 'month', 'days', 'item_shop_first_sale', 'item_first_sale']
num_features = [item for item in X_train.columns if item not in cat_features]

import lightgbm as lgb
# from lightgbm import LGBMRegressor
lgb_params = {
               'feature_fraction': 0.5,
               'metric': 'rmse',
               'min_data_in_leaf': 6,
               'min_sum_hessian_in_leaf': 12,
			   'max_bin':250,
			   'max_depth':15,
               'bagging_fraction': 0.9, 
               'learning_rate': 0.0075, 
               'objective': 'rmse', 
               'bagging_seed': 0, 
               'num_leaves': 256,
               'bagging_freq': 1,
               'verbose_eval': 25,
			   'num_threads': 16,
			   'categorical_feature': cat_features,
			   'n_estimators': 5000
              }

lgb_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=Y_train, categorical_feature=cat_features))
pred_lgb = lgb_model.predict(X_test)

X_train[cat_features] = X_train[cat_features].astype(object)
X_train[num_features] = X_train[num_features].astype(np.float32)
Y_train = Y_train.astype(np.float32)
X_test[cat_features] = X_test[cat_features].astype(object)
X_test[num_features] = X_test[num_features].astype(np.float32)

# %% [code]
from catboost import CatBoostRegressor

cb_model = CatBoostRegressor(
    iterations = 5000, 
    boosting_type = 'Ordered',
    depth = 4,
    max_ctr_complexity = 3,
    bootstrap_type='Bernoulli'
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
X_test_level2 = np.c_[pred_gb, pred_cb, pred_rf, pred_lgb]

eval_month_set = [22, 31, 32, 33]

# That is how we get target for the 2nd level dataset
Y_train_level2 = matrix.loc[matrix['date_block_num'].isin(eval_month_set), 'item_cnt_month']

# dates_train_level2 = dates_train[dates_train.isin([22, 33])]

# And here we create 2nd level feeature matrix, init it with zeros first
# X_train_level2 = np.zeros([Y_train_level2.shape[0], 3])

# Now fill `X_train_level2` with metafeatures

iter = 0
for cur_block_num in eval_month_set:
    
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
    
    X_train = matrix.loc[(matrix['date_block_num'] < cur_block_num) & (matrix['date_block_num'] > (cur_block_num-19))].drop(['item_cnt_month'], axis=1)
    Y_train = matrix.loc[(matrix['date_block_num'] < cur_block_num) & (matrix['date_block_num'] > (cur_block_num-19)), 'item_cnt_month']
    X_test = matrix.loc[matrix['date_block_num'] == cur_block_num].drop(['item_cnt_month'], axis=1)
    
    #xgb
    
    model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train)], 
    verbose=True)
    pred_gb = model.predict(X_test).clip(0,20)
	
	#lgb

    lgb_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=Y_train), categorical_feature=cat_features)
    pred_lgb = lgb_model.predict(X_test)
    
    #cb
    
    X_train.drop('date_block_num', axis=1, inplace=True)
    X_test.drop('date_block_num', axis=1, inplace=True)

    X_train[cat_features] = X_train[cat_features].astype(object)
    X_train[num_features] = X_train[num_features].astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test[cat_features] = X_test[cat_features].astype(object)
    X_test[num_features] = X_test[num_features].astype(np.float32)
    
    cb_model.fit(
    X_train, Y_train,
    verbose = True)
    pred_cb = cb_model.predict(X_test).clip(0,20)
	
	#rf 
	
    rf.fit(X_train, Y_train)
    pred_rf = rf.predict(X_test.values).clip(0,20) 
	
    if iter==0:
	    X_train_level2 = np.c_[pred_gb, pred_cb, pred_rf, pred_lgb]
    else:
	    X_train_level2 = np.vstack((X_train_level2,np.c_[pred_gb, pred_cb, pred_rf, pred_lgb]))
		
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
submission.to_csv('short_stack_submission_v8_2_7_1_3.csv', index=False)

print('model weights',
          'xgb',
          '{:.1%}'.format(meta_model.coef_[0]),
          'cb',
          '{:.1%}'.format(meta_model.coef_[1]),
          'rf',
          '{:.1%}'.format(meta_model.coef_[2]),
		  'lgbm',
          '{:.1%}'.format(meta_model.coef_[3]))


# save predictions for an ensemble
# pickle.dump(Y_test, open('catboost_test.pickle', 'wb'))
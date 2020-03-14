import os
from tqdm import tqdm
import joblib
from joblib import Parallel,delayed
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,classification_report,f1_score
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings('ignore')
import feets
input_dir="/tcdata/"
file=os.listdir(input_dir+'hy_round2_train_20200225')
test_file = os.listdir(input_dir+'hy_round2_testB_20200312')

#特征提取函数
def get_time(df):
    df['time'] = pd.to_datetime(df['time'],format='%m%d %H:%M:%S')
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['min'] = df['time'].dt.minute
    df['time1'] = (np.int64(df['hour']) * 60 + np.int64(df['min'])+np.int64(df['day']*1440))
    beaufort = [(0, 5, 18), (1,0, 5), (1, 17,24)]
    for item in beaufort:
        df.loc[(df['hour']>=item[1]) & (df['hour']<item[2]), 'hour_scale'] = item[0]
    df = df.sort_values('time').reset_index(drop=True)
    return df
def get_a(df):
    b = df['sudu'].diff(1)
    c = df['time1'].diff(1)
    df['a'] = b/(c+0.0001)
    df['a'].fillna(0,inplace=True)
    return df
def clean_sudu(df):
    df['sudu'] = df['sudu'].apply(lambda x: x if x<30 else 30)
    return df
def get_gangkou(df):
#新港口特征
    df['lat_mean'] = df['lat'].mean()
    df['lat_class'] = 0
    df.loc[(df['lat_mean']<20)&(df['lat_mean']>10) , 'lat_class'] = 1
    df.loc[(df['lat_mean']>20)&(df['lat_mean']<26) , 'lat_class'] = 2
    df.loc[(df['lat_mean']>26)&(df['lat_mean']<33) , 'lat_class'] = 3
    df.loc[df['lat_mean']>33 , 'lat_class'] = 4

    df['base_lat'] = 0 ; df['base_lon'] = 0
    df.loc[df['lat_class']==1,'base_lat'] = 18.3
    df.loc[df['lat_class']==2,'base_lat'] = 25.9
    df.loc[df['lat_class']==3,'base_lat'] = 27.1
    df.loc[df['lat_class']==4,'base_lat'] = 36.8

    df.loc[df['lat_class']==1,'base_lon'] = 109.1
    df.loc[df['lat_class']==2,'base_lon'] = 119.6
    df.loc[df['lat_class']==3,'base_lon'] = 120.4
    df.loc[df['lat_class']==4,'base_lon'] = 122.3
    df['lat_base_dis'] = df['lat']-df['base_lat']
    df['lon_base_dis'] = df['lon']-df['base_lon']
    df['lat_base_dis'] = df['lat_base_dis'].abs()
    df['lon_base_dis'] = df['lon_base_dis'].abs()
    df['base_dis'] = ((df['lat_base_dis']**2)+(df['lon_base_dis']**2))**0.5
    df.drop(['lat_mean','lat_class','base_lat','base_lon'],axis=1,inplace=True)
    return df

def cross_num_fea(df,group_col,stat_col):
    for i in group_col:
        temp = df.groupby([i])
        for j in stat_col:
            if i==j:
                continue
            df = df.merge(temp[j].agg({'{}_{}_mean'.format(i,j):'mean','{}_{}_min'.format(i,j): 'min','{}_{}_median'.format(i,j): 'median',
            '{}_{}_max'.format(i,j) :'max','{}_{}_std'.format(i,j):'std'}),on=i,how='left')
    return df        
def soreoccurring(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)
def medianAbsDev(data):
    mad = np.median(np.abs(data-np.median(data)))
    return mad
def get_feet(data,fea):
    t=[]
    for i in fea:
        fs = feets.FeatureSpace(only = ['Amplitude',#'Con', # 'Autocor_length',
                        'Gskew',
                       # 'Meanvariance',
                  # 'MedianAbsDev', 'MedianBRP',
                        'PairSlopeTrend',
                   # 'Autocor_length',
                   # 'AndersonDarling',
                        'Q31',
                        'Rcs',
                   'SmallKurtosis'])
        _, values = fs.extract(magnitude=data[i])
        a = dict(zip(_,values))
        a = pd.DataFrame(a,index=[0])
        a.columns =[i+'_'+j for j in a.columns]
        t.append(a)
        rst = t[0]
        for val in t[1:]:
            rst = pd.concat([rst,val],axis=1)
            rst.reset_index(drop=True, inplace=True)
    return rst
def stat(data,c,name):
    c[name + '_max'] = data.max()
    c[name + '_min'] = data.min()
    c[name + '_mean'] = data.mean()
    c[name + '_ptp'] = data.ptp()
    c[name + '_std'] = data.std()
    c[name + '_skewness'] = data.skew()
    c[name + '_kurtosis'] = data.kurtosis()
    c[name+ '_0.25']=data.quantile(0.25)
    c[name+ '_0.5']=data.quantile(0.5)
    c[name+ '_0.75']=data.quantile(0.75)
    c[name+'_high5']=data.mean()+5*data.std()
    c[name+'_high3']=data.mean()+3*data.std()
    c[name+'_low5']=data.mean()-5*data.std()
    c[name+'_low3']=data.mean()-3*data.std()
    c[name+'_soreoccurring']=soreoccurring(data)
    c[name+'_mode']=data.mode()
 #   c[name+'_medianBRP'] = medianbrp(data)
    c[name+'_medianAbsDev']=medianAbsDev(data)
#    c[name+'_beyond1_std']=beyond1_std(data)
    return c

def stat1(data,c,name):
    c[name + '_max'] = data.max()
    c[name + '_min'] = data.min()
    c[name + '_mean'] = data.mean()
    c[name + '_ptp'] = data.ptp()
    c[name + '_std'] = data.std()
    c[name+ '_0.25']=data.quantile(0.25)
    c[name+ '_0.5']=data.quantile(0.5)
    c[name+ '_0.75']=data.quantile(0.75)
    c[name+'_high5']=data.mean()+5*data.std()
    c[name+'_high3']=data.mean()+3*data.std()
    c[name+'_low5']=data.mean()-5*data.std()
    c[name+'_low3']=data.mean()-3*data.std()
    c[name+'_soreoccurring']=soreoccurring(data)
#    c[name+'_medianBRP'] = medianbrp(data)
    c[name+'_medianAbsDev']=medianAbsDev(data)
 #   c[name+'datapoint']=datapoint(data)
#    c[name+'_beyond1_std']=beyond1_std(data)
    return c
def get_fea(df):
    df['lat_max_lat_min'] = df['lat_max'] - df['lat_min']
    df['lon_max_lon_min'] = df['lon_max'] - df['lon_min']
    df['lon_max_lat_min'] = df['lon_max'] - df['lat_min']
    df['lat_max_lon_min'] = df['lat_max'] - df['lon_min']
    df['slope'] = df['lon_max_lon_min'] / np.where(df['lat_max_lat_min']==0, 0.001, df['lat_max_lat_min'])
    df['area'] = df['lat_max_lat_min'] * df['lon_max_lon_min']
    df['lon_mode_lat_mode'] = df['lon_mode'] - df['lat_mode']
    df['lon_mode*lat_mode'] = df['lon_mode'] * df['lat_mode']
    df['lon_mode/lat_mode'] = df['lon_mode'] / df['lat_mode']
    df['lat_mode/lon_mode'] = df['lat_mode'] / df['lon_mode']
    df['lat_max_lat_mode'] = df['lat_max'] - df['lat_mode']
    df['lon_max_lat_mode'] = df['lon_max'] - df['lat_mode']
    df['lat_max_lon_mode'] = df['lat_max'] - df['lon_mode']
    df['lon_max_lon_mode'] = df['lon_max'] - df['lon_mode']
    df['lat_min_lat_mode'] = df['lat_mode'] -df['lat_min']
    df['lon_min_lat_mode'] =  df['lat_mode']-df['lon_min'] 
    df['lat_min_lon_mode'] = df['lon_mode'] -df['lat_min']
    df['lon_min_lon_mode'] =  df['lon_mode']- df['lon_min']
    return df
#获得训练数据

def basic(idx):
    rst=[]
    data= pd.read_csv(input_dir+'hy_round2_train_20200225/'+file[idx])
    data.rename(columns={'速度':'sudu','方向':'fangxiang'},inplace =True)
    data = get_time(data)
    data = clean_sudu(data)
    data = get_gangkou(data)
    data = get_a(data)
 #   data = get_diff(data,['lon','lat','sudu','fangxiang'])
 #   data =cate_fea(data,['lon','lat','sudu','fangxiang','a'])
 #   data = cross_cate_fea(data,['hour','min'],['lon','lat','sudu','fangxiang','a'])#88有
    data = cross_num_fea(data,['hour','min'],['lon','lat','sudu','fangxiang','a'])#88有
    name = ['lon','lat','sudu','fangxiang']
    name1 = [col for col in data.columns if col not in ['lon','lat','sudu','fangxiang','hour','min','day','time','time1','渔船ID','type','hour_scale']]
    for ti in data.day.unique():
        data1=data[data['day']==ti]
        if len(data1)>12:
            feet = get_feet(data1,['lon','lat','sudu','fangxiang'])
            c= {'ID': data1['渔船ID'].unique(),
            # 'sudu_bi':data1[data1['sudu']<0.0002].shape[0]/data1.shape[0],
            # 'stay_ratiao_y' : data1['lon'].agg(lambda x : x.value_counts().iloc[0]) / data1['lon'].count(),
                'day':300,
                'label':data1['type'].unique(),
             #   'length':len(data1),
          #  'lon_bi':data1['lon'].agg(lambda x:get_mode_count(x)),
          #  'lat_bi':data1['lat'].agg(lambda x:get_mode_count(x)),
            }              
            for j in name1:    
                 c=stat1(data1[j],c,j)
            for j in name:
                c = stat(data1[j],c,j)
            this_tv_features = pd.DataFrame(c, index=[0])
            this_tv_features = get_fea(this_tv_features)
            this_tv_features = pd.concat([this_tv_features,feet],axis= 1)
            rst.append(this_tv_features)
    data2 = data
    feet = get_feet(data2,['lon','lat','sudu','fangxiang'])
    c= {'ID': data2['渔船ID'].unique(),
         #'sudu_bi':data2[data2['sudu']<0.0002].shape[0]/data2.shape[0],
        # 'stay_ratiao_y' : data2['lon'].agg(lambda x : x.value_counts().iloc[0]) / data2['lon'].count(),
        'day':200,
      'label':data2['type'].unique(),
        #'length':len(data2),
         #   'lon_bi':data2['lon'].agg(lambda x:get_mode_count(x)),
         #   'lat_bi':data2['lat'].agg(lambda x:get_mode_count(x)),
    }
    for j in name1:    
        c=stat1(data2[j],c,j)
    for j in name:
        c = stat(data2[j],c,j)
    this_tv_features = pd.DataFrame(c, index=[0])
    this_tv_features = get_fea(this_tv_features)
    this_tv_features = pd.concat([this_tv_features,feet],axis= 1)
    rst.append(this_tv_features)
    tv_features=rst[0]
    for val in rst[1:]:
        tv_features = pd.concat([tv_features,val],axis=0)
        tv_features.reset_index(drop=True, inplace=True)
    return tv_features
ships = Parallel(n_jobs=-1, verbose=10)(delayed(basic)(i) for i in range(len(file)))
train = pd.concat(ships, axis=0)
train.reset_index(drop=True, inplace=True)

#获得测试数据
def basic(idx):
    rst=[]
    data= pd.read_csv(input_dir+'hy_round2_testB_20200312/'+test_file[idx])
    data.rename(columns={'速度':'sudu','方向':'fangxiang'},inplace =True)
    data = get_time(data)
    data = clean_sudu(data)
    data = get_gangkou(data)
    data = get_a(data)
#    data = get_diff(data,['lon','lat','sudu','fangxiang'])
#    data =cate_fea(data,['lat','lon','sudu','fangxiang','a'])
#    data = cross_cate_fea(data,['hour','min'],['lat','lon','sudu','fangxiang','a'])#88有
    data = cross_num_fea(data,['hour','min'],['lat','lon','sudu','fangxiang','a'])#88有
    name = ['lat','lon','sudu','fangxiang']
    name1 = [col for col in data.columns if col not in ['lat','lon','sudu','fangxiang','hour','min','day','time','time1','渔船ID','type','hour_scale']]
    for ti in data.day.unique():
        data1=data[data['day']==ti]
        if len(data1)>12:
            feet = get_feet(data1,['lon','lat','sudu','fangxiang'])
            c= {'ID': data1['渔船ID'].unique(),
                'day':300,
              #  'length':len(data1),
            #'lon_bi':data1['lon'].agg(lambda x:get_mode_count(x)),
           # 'lat_bi':data1['lat'].agg(lambda x:get_mode_count(x)),
               # 'sudu_bi':data1[data1['sudu']<0.0002].shape[0]/data1.shape[0],
                # 'stay_ratiao_y' : data1['lon'].agg(lambda x : x.value_counts().iloc[0]) / data1['lon'].count(),
               # 'label':data1['type'].unique()
            }              
            for j in name1:    
                 c=stat1(data1[j],c,j)
            for j in name:
                c = stat(data1[j],c,j)
            this_tv_features = pd.DataFrame(c, index=[0])
            this_tv_features = get_fea(this_tv_features)
            this_tv_features = pd.concat([this_tv_features,feet],axis= 1)
            rst.append(this_tv_features)
    data2 = data
    feet = get_feet(data2,['lon','lat','sudu','fangxiang'])
    c= {'ID': data2['渔船ID'].unique(),
      'day':200,
       # 'length':len(data2),
    # 'lon_bi':data2['lon'].agg(lambda x:get_mode_count(x)),
   # 'lat_bi':data2['lat'].agg(lambda x:get_mode_count(x)),
      #'sudu_bi':data2[data2['sudu']<0.0002].shape[0]/data2.shape[0],
       # 'stay_ratiao_y' : data2['lon'].agg(lambda x : x.value_counts().iloc[0]) / data2['lon'].count(),
     # 'label':data2['type'].unique()
    }
    for j in name1:    
        c=stat1(data2[j],c,j)
    for j in name:
        c = stat(data2[j],c,j)
    this_tv_features = pd.DataFrame(c, index=[0])
    this_tv_features = get_fea(this_tv_features)
    this_tv_features = pd.concat([this_tv_features,feet],axis= 1)
    rst.append(this_tv_features)
    tv_features=rst[0]
    for val in rst[1:]:
        tv_features = pd.concat([tv_features,val],axis=0)
        tv_features.reset_index(drop=True, inplace=True)
    return tv_features
ships = Parallel(n_jobs=-1, verbose=10)(delayed(basic)(i) for i in range(len(test_file)))
test = pd.concat(ships, axis=0)
test.reset_index(drop=True, inplace=True)

#模型建立
dic = {'围网':0,'拖网':1,'刺网':2}
train['label'] = train['label'].map(dic)
feature_names = [i for i in test.columns if i not in ['ID','label','day'] ]
X_train = train[feature_names]
y = train['label']
X_test = test[feature_names]
print(X_train.shape)
#特征筛选
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
#use linear regression as the model
X_train = impute(X_train)
lr = RandomForestClassifier()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=840)
rfe.fit(X_train,y)
c = pd.DataFrame()
c['fea'] = feature_names
c['score'] = c['fea'].map(dict(zip(feature_names,rfe.ranking_)))
c = c.sort_values('score',ascending=False)
fea = c[c.score ==1]['fea']
X_train = train[fea]
X_test = test[fea]
#五个种子十折LGB
params = {
    'n_estimators': 5000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'early_stopping_rounds': 100
}
seeds=[520,2020,118,720,815]
num_model_seed=3
models = []
pred = np.zeros((len(X_test),3))
oof = np.zeros((len(X_train), 3))

for model_seed in range(num_model_seed):
    print(model_seed + 1)
    oof_cat = np.zeros((len(X_train), 3))
    prediction_cat = np.zeros((len(X_test),3))
    fold = StratifiedKFold(n_splits=8, random_state=seeds[model_seed], shuffle=True)
    
    for index, (train_idx, val_idx) in enumerate(fold.split(X_train, y)):
        train_set = lgb.Dataset(X_train.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X_train.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)
        models.append(model)
        val_pred_cat = model.predict(X_train.iloc[val_idx])
        
        oof_cat[val_idx] += val_pred_cat
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred_cat, axis=1)
        print(index, 'val f1',f1_score(val_y, val_pred, average='macro'))
        test_pred = model.predict(X_test)
        prediction_cat += test_pred/8       
    oof += oof_cat / num_model_seed
    pred += prediction_cat / num_model_seed
oof_ = np.argmax(oof, axis=1)
print('oof f1', f1_score(oof_, y, average='macro'))

#得到结果
v_dict ={0:'围网',1:'拖网',2:'刺网'}
re = pd.DataFrame()
re['ID'] = test['ID']
re['label'] = np.argmax(pred,axis=1)
re['pre_label'] = re.groupby(['ID'])['label'].transform(lambda x : x.mode()[0])
re = re.drop_duplicates('ID').reset_index(drop=True)
sub = re[['ID','pre_label']]
sub['pre_label'] = sub.pre_label.map(v_dict)
sub = sub.sort_values('ID')
sub.to_csv(("result.csv"), header=None, index=False)
print(sub['pre_label'].value_counts())

import math
import pickle
import inflection
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler

class Rossmann(object):
    def __init__(self):
        self.home_path = 'C:\\Users\\prove\\rossmann-store-sales\\'
        self.competition_distance_scaler = pickle.load(open(self.home_path + 'parameter\\competition_distance_scaler.p', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter\\competition_time_month_scaler.p', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(self.home_path + 'parameter\\promo_time_week_scaler.p', 'rb'))
        self.year_scaler= pickle.load(open(self.home_path + 'parameter\\year_scaler.p', 'rb'))
        self.store_type_encoding= pickle.load(open(self.home_path + 'parameter\\store_type_encoding.p', 'rb'))
        self.model = pickle.load(open(self.home_path + "models\\xgb_model.p", "rb" ))
    
    def data_cleaning(self, df1):
        cols_old = list(df1.columns)
        cols_new = map(lambda x: inflection.underscore(x), cols_old)
        df1.columns = cols_new
        print(df1.columns)
        #df1 = df1.drop(columns=['sales', 'customers'])
        
        ### 1.2 Data Dimensions

        print("# of rows: {}\n# of columns: {}".format(*df1.shape))

        ### 1.3 Data Types

        df1['date'] = pd.to_datetime(df1['date'])

        ## 1.5 Fillout NA

        # competition_distance - distance in meters to the nearest competitor store
        # There is no competitor nearby, therefore we can assume that NAN fields can be filled with a large number.
        df1['competition_distance'] = df1['competition_distance'].astype('float64')
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        #competition_open_since_mouth - gives the approximate year and month of the time the nearest competitor was opened
        # There is no competitor nearby or we don't know when it opened. We are assuming the date of the last sell.
        print(df1['date'].dtype)
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)
        
        #competition_open_since_year 
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        #promo2_since_week - describes the year and calendar week when the store started participating in Promo2
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        #promo2_since_year 
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)
        
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        #promo_interval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
        month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dez'}
        df1['promo_interval'].fillna(0, inplace=True)
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
    
        return df1

    def feature_engineering(self, df2):
        #date
        df2['year'] = df2['date'].dt.year
        df2['month'] = df2['date'].dt.month
        df2['day'] = df2['date'].dt.day
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        #competition since
        df2['competition_since'] = df2.apply(lambda x: datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)

        #promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.strptime(x+'-1', '%Y-%W-%w') - timedelta(days=7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)

        #assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        #state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x =='b' else 'christmas' if x == 'c' else 'regular_day')

        # 3.0 Variable Filtering
        ## 3.1 Rows Filtering
        df2 = df2[(df2['open'] != 0)]

        ## 3.2 Variable Selection
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)

        ## 1.6 Change Data Types
        df2['competition_open_since_month'] = df2['competition_open_since_month'].astype(int)
        df2['competition_open_since_year'] = df2['competition_open_since_year'].astype(int)

        df2['promo2_since_week'] = df2['promo2_since_week'].astype(int)
        df2['promo2_since_year'] = df2['promo2_since_year'].astype(int)
        
        return df2

    def data_preparation(self, df5):
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)
        #pickle.dump(rs, open('parameter\\competition_distance_scaler.p', 'wb'))

        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values)
        #pickle.dump(rs, open('parameter\\competition_time_month_scaler.p', 'wb'))

        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)
        #pickle.dump(rs, open('parameter\\promo_time_week_scaler.p', 'wb'))

        df5['year'] = self.competition_distance_scaler.fit_transform(df5[['year']].values)
        #pickle.dump(rs, open('parameter\\year_scaler.p', 'wb'))

        ### 5.2 Transformations
        ### 5.2.1 Encoding

        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        le = LabelEncoder()
        df5['store_type'] = le.fit_transform(df5['store_type'])

        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        ### 5.2.2 Target variable transformation

        #df5['sales'] = np.log1p(df5['sales'])

        ### 5.2.3 Cyclical variables

        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))

        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))

        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))

        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/52)))

        cols_selected = ['store', 'promo', 'store_type', 'assortment',
                         'competition_distance', 'competition_open_since_month',
                         'competition_open_since_year', 'promo2', 'promo2_since_week',
                         'promo2_since_year', 'competition_time_month', 'promo_time_week',
                         'day_of_week_sin', 'day_of_week_cos', 'month_cos',
                         'day_sin', 'day_cos', 'week_of_year_cos']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)
        original_data['prediction'] = np.expm1(pred)
        return original_data.to_json(orient='records', date_format='iso')
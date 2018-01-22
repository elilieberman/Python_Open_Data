# -*- coding: utf-8 -*-
"""
NYC Open Data, Department of Health, Restaurant Violations
https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data
** Goal was to tidy the data, and learn if certain types of restaurants and violations, had a high correlation to closure, why and think about what can be done to mitigate.

"""
# An inspection score of 0-13 is an A, 14-27 points is a B, and 28 or more points is a C
# http://www1.nyc.gov/site/doh/business/food-operators/letter-grading-for-restaurants.page
# Import JSON data from NYC Open data site and convert to data frame
# Socrata end-point "9w7m-hzhe"

import pandas as pd
pd.set_option('display.max_columns', None)
pd.option_context('display.max_colwidth', None)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sodapy import Socrata #to directly grab csv from NYC open data
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE

#%% Retrieve latest data from NYC open-data site
client = Socrata("data.cityofnewyork.us", None)
results = client.get("9w7m-hzhe", limit=1000000)
eats = pd.DataFrame.from_records(results) # Convert to pandas DataFrame, NaN are introduced in the process
eats.isnull().sum().sort_values(ascending = False).head() #summary of missing values by column/feature
#%% Tidy dataframe, create date objects and binary features for later classification tasks
eats['food'] = eats.cuisine_description.apply(lambda x: x[:5])
eats['rec_date'] = eats['record_date'].apply(lambda x: datetime.datetime.strptime(x.split('T')[0], '%Y-%m-%d'))
eats['ins_date'] = eats['inspection_date'].apply(lambda x: datetime.datetime.strptime(x.split('T')[0], '%Y-%m-%d'))
eats.grade_date.fillna('1948-05-14T00:00:00.000', inplace = True) # 1948-05-14 dummy value for missing grade dates for strftime function to work
eats['grd_date'] = eats['grade_date'].apply(lambda x: datetime.datetime.strptime(x.split('T')[0], '%Y-%m-%d'))
eats['crt_flag'] = eats['critical_flag'].apply(lambda x: 1 if x == "Critical" else 0)
eats['action'] = eats['action'].apply(str)
eats['closed'] = eats['action'].apply(lambda x: 1 if 'Closed' in x  else 0) # creat classification feature
eats.inspection_type.fillna('1948-05-14/1948-05-14', inplace = True) # 1948-05-14 dummy value for split function to work
#eats['inspection_type'] = eats['inspection_type'].astype(str)
eats['ins_a'] = eats['inspection_type'].apply(lambda x: x.split('/')[0])  #breakout of inspection_type
eats['ins_b'] = eats['inspection_type'].apply(lambda x: x.split('/')[-1]) #breakout of inspection_type
eats['score'] = eats.score.fillna(-99999)
eats['score'].apply(int)
eats = eats.sort_values(['camis','ins_date', 'inspection_type'], ascending=[True, True, True])
#%% Create "start" date, the date of Pre-permit (Operational) / Initial Inspection
def s_date (df):
#   if df.inspection_type == 'Pre-permit (Operational) / Initial Inspection':
    if 'Initial Inspection' in df.inspection_type:    
        d = df.ins_date
        return d
    else:
        d = datetime.datetime.strptime('1948-05-14 00:00:00', '%Y-%m-%d %H:%M:%S') #maintain datetime object 
        return d
eats['sdate'] = eats.apply(s_date, axis =1)

#%% prepare table of  first violation date
v_df = eats.loc[eats['violation_code'] != ''] #subset table for violation data, get ONLY violation records
v_df = v_df.loc[:,('camis','ins_date')].sort_values(['camis','ins_date'], ascending=[True, True]) #simplify subset to restaurant and violation dates
v_df.head(25)  # data check
v_df.columns = ['camis', 'vdate'] #rename column to first violation date 'vdate'
v_df = pd.DataFrame(v_df.groupby('camis', as_index = False).first()) #take first row, which is date of first violation
v_df = v_df.join(eats.sdate) #add start date to table
v_df.dtypes #data check
v_df.head(5) #data check

# FEATURE start to violation, and backing out noisy data
v_df['str_viol'] = v_df.apply(lambda x: x['vdate'] - x['sdate'], axis =1) # compute time, START to FIRST violation was recorded (post opening day)
v_df['str_viol'] = v_df.apply(lambda x: datetime.timedelta(days=0) if x.sdate == datetime.datetime.strptime('1948-05-14 00:00:00', '%Y-%m-%d %H:%M:%S') else x.str_viol, axis =1) # convert dummyvalue/placeholders 05/14/1948 to nan's or 0 depending on model requ
v_df.loc[v_df['str_viol'] < datetime.timedelta(days=0)].count() #check for negative values occuring from the dummy dates
v_df['str_viol'] = v_df.apply(lambda x: datetime.timedelta(days=0) if x.str_viol < datetime.timedelta(days=0) else x.str_viol, axis =1) # convert negative dates to nan's or ), depending on the model need
v_df['sv_days'] = v_df['str_viol'].apply(lambda x: x/np.timedelta64(1, 'D')).astype(int)#convert timedelta days to intergers for analysis
v_df.loc[:,['camis', 'str_viol'] ] #data check
#%% flatten violation_code using dummy columns, drop original column
eats.violation_code.value_counts()
viol = pd.get_dummies(eats['violation_code'], prefix = 'v')
viol.head()
viol = viol.join(eats.camis) #add camis field for grouping
# FEATURE count of violations
viol = pd.DataFrame(viol.groupby('camis',as_index = False ).sum()) 

# Breakout cuisine type for regression, groupby LAST to flatten table for grouping by camis 
eats.cuisine_description.value_counts()
cuisine = pd.get_dummies(eats['food'], prefix = 'f')
cuisine.head()
cuisine = cuisine.join(eats.camis) #add camis field to table for grouping
cuisine = cuisine.join(eats.closed)
cuisine = pd.DataFrame(cuisine.groupby('camis', as_index = False).last())

# merge violation and dummy features
cuis_viol = pd.merge(cuisine, viol, on='camis', how='left')
cuis_viol.head()

#%% Pandas Dataframe versus Numpy Array
X = cuis_viol.drop(['closed','camis'],1)
y = cuis_viol['closed']
#X = preprocessing.scale(X)
cuis_viol.dropna(inplace = True)
y = cuis_viol['closed']
print(len(X),len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

#%%

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)# pulled from above
# Use feature selection to select the most important features
import sklearn.feature_selection
select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]
X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]
print(colnames_selected)


#Recursive Feature Elimination
model = LogisticRegression()
clf = RFE(model, 10) #feature reduction
clf.fit(X_train_selected,y_train) #fit the model using the training data
accuracy = clf.score(X_test_selected, y_test) # test the fit 
# summarize the selection of the attributes
print(clf.support_)
print(clf.ranking_)

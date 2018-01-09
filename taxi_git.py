# -*- coding: utf-8 -*-
"""
Script to generate chart of New York City ride-share major competitors market-share
"""
import pandas as pd
pd.set_option('display.max_columns', None)
pd.option_context('display.max_colwidth', None)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sodapy import Socrata #to directly grab csv from NYC open data

client = Socrata("data.cityofnewyork.us", None)
results = client.get("edp9-qgv4", limit=100000)
# Convert to pandas DataFrame
fhv = pd.DataFrame.from_records(results)


#%% digitize date fields, features
fhv['pickup_start_date'] = fhv['pickup_start_date'].apply(lambda x: datetime.datetime.strptime(x.split('T')[0], '%Y-%m-%d'))
fhv['pickup_end_date'] = fhv['pickup_end_date'].apply(lambda x: datetime.datetime.strptime(x.split('T')[0], '%Y-%m-%d'))
fhv['start_year'] = fhv['pickup_start_date'].apply(lambda x: x.year)
fhv['start_month'] = fhv['pickup_start_date'].apply(lambda x: x.month)
fhv['period'] = fhv['pickup_start_date'].apply(lambda x: x.strftime('%Y-%m'))  
#%% turn ambigious OBJECTS in to strings and numerics for calculations
fhv['dba'] = fhv['dba'].apply(str)       
fhv['base_name'] = fhv['base_name'].apply(str)
fhv['trips'] = fhv['total_dispatched_trips'].astype(int)
#%% basic data descriptors
fhv.info()
fhv.describe(include = 'all')

#%% group by dba i.e. to ID major operators
fhv['base_name'].value_counts().sort_values(ascending = False)
fhv['dba'].value_counts().sort_values(ascending = False)

#%% setup aggregation of major operators, Uber, Via, Gett, Lyft, Juno, and related DBA's
def short_name (df):
    name = df['dba']
    top_name = df['base_name']
    
    if  'UBER' in name:
        return 'UBER'
    elif 'JUNO' in name:
        return 'UBER'  #Juno is now UBER
    elif 'VIA' in name:
        return 'VIA'
    elif 'LYFT' in name:
        return 'LYFT'
    elif 'GETT' in name:
        return 'GETT'     
    elif 'nan' in name:
        return top_name
    else:
        return (name)
    
# review 5 major operator counts
fhv['dba_short'] = fhv.apply(short_name, axis = 1)
fhv['dba_short'].value_counts().sort_values(ascending = False)
biglist = ['UBER', 'VIA', 'LYFT', 'GETT']
biglist
uglv = fhv.loc[fhv['dba_short'].isin(biglist)]
rest = fhv.loc[fhv['dba_short'].isin(biglist)==False]
fhv['company'] = fhv.apply(lambda x: x.dba_short if (x.dba_short in biglist)  else 'OTHER', axis = 1) # GroupBy helper, the big 4 versus all other, with 'ternary' a if condition else b
fhv.head()
fhv.tail()
#similar  other = fhv.loc[~fhv['dba_short'].isin(biglist)]

fhv2 = fhv[['period', 'company', 'trips']].copy()
fhv2['company'].unique()  
fhv2['company'].value_counts()#value_counts
fhv2.pivot_table(index = 'period', columns = 'company', values = [ 'trips'], aggfunc = 'sum', fill_value = 0)
fhv2.pivot_table(index = 'period', columns = 'company', values = [ 'trips'], aggfunc = 'sum', fill_value = 0).plot.bar( figsize = (12,9), align = 'center',  width = 2, title = 'NYC Ride Share Major Competitors, FHV Trip Volume in Millions')

#%%
plt.savefig('NYC_Ride_Share_Competition.png')  # save image to local directory 
fhv.to_csv('fhv.csv', encoding = 'utf-8') # save data set with features to csv
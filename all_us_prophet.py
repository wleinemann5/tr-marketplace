#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:47:53 2021

@author: wilhelmleinemann
"""

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn import preprocessing
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import snowflake.connector
from snowflake.connector import DictCursor
import json

with open("snowflake_credentials.txt") as file:
    cred = json.load(file)
    
db_con = snowflake.connector.connect(**cred)
cur = db_con.cursor(DictCursor)

def sqlExec(query):
    df = pd.DataFrame(cur.execute(query).fetchall())
    df.columns = map(str.lower, df.columns)
    return df

def ikea_map(category_name):
    if (category_name == 'IKEA Delivery') or (category_name == 'IKEA Assembly') or (category_name == 'IKEA GM Assembly'):
        return "IKEA"
    else:
        return category_name
        

def add_covid_impact(df):
    df = pd.merge(left=df,
         right=df_covid,
         on='ds',
         how='left')
    return df


def gb_helper(df):
    df = df.groupby(by='ds',as_index=False).sum()
    df = add_covid_impact(df)
    return df


us_sql = """

with recos as
(
select TRUNC(tr.local_created_at,'day') reco_day
  ,    COUNT(DISTINCT tr.recommendation_id) as num_recos
  ,    AVG(CASE WHEN tr.number_of_results > 30 THEN 30 ELSE tr.number_of_results END) as avg_search_results
from "TASKRABBIT_PRODUCTION"."PUBLIC"."JOB_RECOMMENDATIONS" tr
join public.categories c
on tr.category_id = c.id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on tr.metro_id = m.metro_id
where tr.local_created_at >= '2017-12-01'
and tr.last_in_funnel
and tr.locale = 'en-US'
and tr.category_id not in (1104,1106,1107)
--and tr.metro_id in (1060)
group by TRUNC(tr.local_created_at,'day')
order by reco_day
  ,      num_recos DESC
)
,
jobs_select as 
(
select TRUNC(j.opened_state_at,'day') reco_day
  ,    COUNT(DISTINCT j.id) as num_jobs
from jobs as j
join public.categories c
on j.category_id = c.id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on j.metro_id = m.metro_id
where j.opened_state_at >= '2017-12-01'
and j.v2_id is NULL
and j.booking_attempt = 1
and j.fixup_key is NULL
and (j.suppress_level < 1000 or j.suppress_level is NULL)
and j.locale = 'en-US'
and j.category_id not in (1104,1106,1107)
--and j.metro_id in (1060)
group by TRUNC(j.opened_state_at,'day')
order by reco_day
  ,      num_jobs DESC
)
,
invoices as
(
select TRUNC(ji.submitted_state_at,'day') reco_day
  ,    COUNT(DISTINCT ji.id) as num_invoices
  ,    SUM(ji.poster_subtotal_cents) as gmv
from public.job_invoices ji
join public.jobs j
on ji.job_id = j.id
join public.categories c
on j.category_id = c.id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on j.metro_id = m.metro_id
where ji.submitted_state_at >= '2017-12-01'
and j.v2_id is NULL
and j.booking_attempt = 1
and j.fixup_key is NULL
and (j.suppress_level < 1000 or j.suppress_level is NULL)
and j.locale = 'en-US'
and j.category_id not in (1104,1106,1107)
--and j.metro_id in (1060)
group by TRUNC(ji.submitted_state_at,'day')
order by reco_day
  ,      num_invoices DESC
)
,
taskers as 
(
select distinct count(rca.rabbit_id) as num_taskers
  ,    SUM(rca.total_availability_that_day) as total_seconds
  ,    SUM(rca.appointment_seconds_booked) as booked_seconds
  ,    rca.date as reco_day
from RABBIT_CALENDAR_AVAILABILITIES rca
join rabbit_profiles rp
on rca.rabbit_id = rp.user_id
where rca.date >= '2017-12-01'
--and rca.date <= '2021-05-01'
--and rp.geo_json_metro_id in (1063,1078)
group by rca.date
)


select recos.reco_day
  ,    recos.num_recos
  ,    recos.avg_search_results
  ,    jobs_select.num_jobs
  ,    invoices.num_invoices
  ,    invoices.gmv
  ,    taskers.num_taskers
  ,    taskers.total_seconds
  ,    taskers.booked_seconds
from recos
left join jobs_select
on (recos.reco_day = jobs_select.reco_day)
left join invoices
on (recos.reco_day = invoices.reco_day)
left join taskers
on (recos.reco_day = taskers.reco_day)

"""

df = sqlExec(us_sql)

df_covid = pd.read_csv('covid_impact.csv')
df_covid = df_covid.rename(columns={'Name':'ds','United States':'covid_impact'})
df_covid.ds = pd.to_datetime(df_covid.ds)
df_covid = df_covid.filter(items=['ds','covid_impact'])


def simple_model_helper(df,regressor_name):
    m = Prophet()
    m.add_regressor('covid_impact')
    m.add_country_holidays(country_name='US')
    m.add_seasonality(name='monthly',period=30.5, fourier_order=5)
    m.fit(df)
    future_df = m.make_future_dataframe(periods=365)
    future_df = add_covid_impact(future_df)
    forecast_df = m.predict(future_df).filter(items=['ds','yhat'])
    forecast_df['yhat'] = np.clip(forecast_df['yhat'],0,100000)
    forecast_df = forecast_df.rename(columns={'yhat':regressor_name})
    return forecast_df


#main loop
#df['aov'] = df['gmv']/df['num_invoices']/100
target = 'num_invoices'
df_all = df.rename(columns={'reco_day':'ds',target:'y'})
df_all = df_all[df_all['ds'] < '2022-02-27']
df_all = df_all[df_all['ds'] > '2018-10-31']
total_preds = pd.DataFrame()

#met_list = ['London']
#for met in df_merge.metro_name.unique()
#met = 'London'
df_all = gb_helper(df_all)

m = Prophet(seasonality_mode='multiplicative',interval_width=0.95)
m.add_regressor('covid_impact')
m.add_country_holidays(country_name='US')
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(df_all)
pred = m.predict(df_all)
m.plot(pred)
m.plot_components(pred)

df_f = m.make_future_dataframe(periods=365)
df_f = add_covid_impact(df_f)
df_forecast = m.predict(df_f)
#df_forecast.to_csv('invoice_forecast_1116.csv',index=False)
m.plot(df_forecast)
m.plot_components(df_forecast)
df_forecast['yhat'] = np.clip(df_forecast['yhat'],0,100000)
#df_forecast['metro_name'] = met
df_forecast = df_forecast.filter(items=['ds','yhat'])
df_forecast = pd.merge(left = df_all,
               right = df_forecast,
               on = 'ds',
               how = 'right')
total_preds = pd.concat([total_preds,df_forecast],ignore_index=True)


total_preds = total_preds.rename(columns={'y':target,'yhat':'pred_'+target})

####################
#####stop here#####
####################

reco_preds = total_preds[['ds','pred_num_recos']]
job_preds = total_preds[['ds','pred_num_jobs']]
invoice_preds = total_preds[['ds','pred_num_invoices']]
aov_preds = total_preds[['ds','pred_aov']]

df_forecast.to_csv('invoice_forecast_1115.csv',index=False)

df_all = df_all.merge(right = reco_preds, on = 'ds', how='outer').merge(right= job_preds, on = 'ds').merge(right=invoice_preds, on='ds').merge(right=aov_preds, on='ds')

df_all.to_csv('forecasts_0526.csv',index=False)

total_preds['month_year']= total_preds.ds.dt.to_period('M')
gb = total_preds.groupby('month_year').sum()
gb['pred_MoM']  = (gb.pred_num_invoices/gb.shift(1).pred_num_invoices) -1 
gb['MoM']  = (gb.num_invoices/gb.shift(1).num_invoices) -1 

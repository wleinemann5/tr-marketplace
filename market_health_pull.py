#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:10:50 2021

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

start_date = '2021-06-01'

query_supply = """
select
distinct
jr.recommendation_id,
date_trunc('day', jr.created_at) as day,
date_trunc('week', jr.created_at) as week,
date_trunc('month', jr.created_at) as month,
jr.locale,
metros.name  as metro_name,
case when lower(categories.name) like '%ikea%' then 'IKEA' else categories.name end as cat_name,
case when jobs.id is NULL then 0  else 1 end as job_booked,
case when jr.number_of_results > 30 then 30 else jr.number_of_results end as number_of_results

from job_recommendations jr
LEFT JOIN jobs on jr.recommendation_id = jobs.recommendation_id
LEFT JOIN categories  ON jr.category_id  = categories.id
LEFT JOIN geographies  AS metros ON COALESCE(jr.metro_id, jr.geography_id)  = metros.id

WHERE jr.created_at  >= '{start_date}'
        and UPPER(jr.locale ) = UPPER('en-US')
        and jr.last_in_funnel 
        and (jobs.suppress_level < 1000 or jobs.suppress_level is NULL) 
""".format(start_date = start_date)

df_supply = sqlExec(query_supply)

gb_cols = ['month','cat_name']

#get recos
df_gb = df_supply.groupby(gb_cols, as_index = False).job_booked.count()
df_gb = df_gb.rename(columns = {'job_booked':'recos'})

#get jobs
df_gb = df_gb.merge(df_supply.groupby(gb_cols, as_index = False).job_booked.sum())
df_gb = df_gb.rename(columns = {'job_booked':'jobs'})

#get reco_conversion and market score
df_gb['reco_conversion'] = df_gb.jobs / df_gb.recos
df_gb['market_score'] = df_gb.reco_conversion * np.sqrt(df_gb.recos)

#make l1 df
df_L1 = df_gb[gb_cols + ['recos','reco_conversion','market_score']]
df_L1


#L2s
## Pull price information
query_norm_px = """with norm_px as(
select 
    DISTINCT
    job_recommendations.created_at as date,
    date_trunc('week', job_recommendations.created_at) as week,
    date_trunc('month', job_recommendations.created_at) as month,
    date_trunc('day', job_recommendations.created_at) as day,
    jrr.recommendation_id,
    LEAST(job_recommendations.NUMBER_OF_RESULTS,21) as MAX_POSITION,
    case when jobs.id is NULL then 0  else 1 end as job_booked,
    case when invitations.rabbit_id is NULL then 0 else 1 end as rabbit_booked,
    metros.name  as metro_name,
    case when lower(categories.name) like '%ikea%' then 'IKEA' else categories.name end as cat_name,
    case when job_recommendations.vehicle_requirement = 'truck' then 'truck'
         when job_recommendations.vehicle_requirement = 'car_or_truck' then 'car_or_truck'
         else 'none' end as vehicle_requirement,
    jrr.position,
    to_double(jrr.rabbit_hourly_rate_cents / 100.0) as rabbit_rate,
    ep.category_invoice_count,
    ep.expected_px,
    (to_double(jrr.rabbit_hourly_rate_cents / 100.0) - ep.expected_px)/ep.expected_px as px_error,
    100 * (1+ ((to_double(jrr.rabbit_hourly_rate_cents / 100.0) - ep.expected_px)/ep.expected_px)) as normalized_price    
    
FROM job_recommendation_recommendations jrr 
LEFT JOIN job_recommendations ON job_recommendations.recommendation_id = jrr.recommendation_id
LEFT JOIN categories  ON job_recommendations.category_id  = categories.id
LEFT JOIN geographies  AS metros ON COALESCE(job_recommendations.metro_id, job_recommendations.geography_id)  = metros.id
LEFT JOIN invitations ON jrr.recommendation_id = invitations.recommendation_id AND jrr.user_id = invitations.rabbit_id
LEFT JOIN jobs on jrr.recommendation_id = jobs.recommendation_id
left join playground.bb_expected_price ep on ep.METRO_NAME = metros.name and ep.CAT_NAME = (case when lower(categories.name) like '%ikea%' then 'IKEA' else categories.name end)
    and (case when job_recommendations.vehicle_requirement = 'truck' then 'truck'
         when job_recommendations.vehicle_requirement = 'car_or_truck' then 'car_or_truck'
         else 'none' end ) = (case when ep.vehicle_requirement is NULL then 'none' else ep.vehicle_requirement end)
    and (case when jrr.category_invoice_count > 75 then 75 else jrr.category_invoice_count end) = ep.category_invoice_count

WHERE job_recommendations.created_at  >= '{start_date}' and
         (jrr.position  < 21) AND 
         ((UPPER(job_recommendations.locale ) = UPPER('en-US'))) AND 
          job_recommendations.last_in_funnel and
          (jobs.suppress_level < 1000 or jobs.suppress_level is NULL) 

order by recommendation_id, Position
)

select
norm_px.RECOMMENDATION_ID,
norm_px.date,
norm_px.day,
norm_px.week,
norm_px.month,
norm_px.METRO_NAME,
norm_px.CAT_NAME,
norm_px.VEHICLE_REQUIREMENT,
norm_px.MAX_POSITION,
sum(norm_px.normalized_price * w.WEIGHT) as price_shown_idx,
sum(norm_px.rabbit_rate * w.WEIGHT) as wt_price_shown,
sum(norm_px.category_invoice_count * w.WEIGHT) as wt_cat_experience,
max(norm_px.job_booked) as JOB_BOOKED
from

norm_px 
left join playground.bb_pos_weights w on w.MAX_POSITION = norm_px.MAX_POSITION and w.POSITION = norm_px.POSITION

group by 1,2,3,4,5,6,7,8,9
order by 2
""".format(start_date = start_date)

#########################################
## pull recos/taskers by metro         ##
## You will need to adjust this query  ##
##  if you want to group by categories ##
#########################################

query_rpt_metro = """
with recos as
(
select tr.metro_id
  ,    m.name as metro_name
  ,    TRUNC(tr.local_created_at,'day') reco_day
  ,    TRUNC(tr.local_created_at,'month') reco_month
  ,    TRUNC(tr.local_created_at,'week') reco_week
  ,    COUNT(DISTINCT tr.recommendation_id)as num_recos
  
from "TASKRABBIT_PRODUCTION"."PUBLIC"."JOB_RECOMMENDATIONS" tr
join public.categories c
on tr.category_id = c.id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on tr.metro_id = m.metro_id
where tr.local_created_at >= '{start_date}'
and tr.last_in_funnel
and tr.locale = 'en-US'
group by 1,2,3,4,5
)

,
taskers as 
(
select count(distinct rca.rabbit_id) as num_taskers
  ,    TRUNC(rca.date,'day') as reco_day
  ,    rp.geo_json_metro_id as metro_id
  ,    m.name as metro_name
from RABBIT_CALENDAR_AVAILABILITIES rca
join rabbit_profiles rp
on rca.rabbit_id = rp.user_id
join rabbit_categories rc
on rca.rabbit_id = rc.rabbit_id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on rp.geo_json_metro_id = m.metro_id
where rca.date >= '{start_date}'
and rca.locale = 'en-US'
and rca.TOTAL_AVAILABILITY_THAT_DAY > 0
group by rca.date
  ,  rp.geo_json_metro_id
  ,  m.name
)
select recos.metro_id
  ,    recos.metro_name
  ,    recos.reco_day as DAY
  ,    recos.reco_week as WEEK
  ,    recos.reco_month as MONTH
  ,    recos.num_recos
  ,    taskers.num_taskers
from recos
left join taskers
on (recos.metro_id = taskers.metro_id) and (recos.metro_name = taskers.metro_name) and (recos.reco_day = taskers.reco_day) 
""".format(start_date = start_date)

query_rpt_category = """
with recos as
(
select tr.category_id
  ,    c.name as cat_name
  ,    TRUNC(tr.local_created_at,'day') reco_day
  ,    TRUNC(tr.local_created_at,'month') reco_month
  ,    TRUNC(tr.local_created_at,'week') reco_week
  ,    COUNT(DISTINCT tr.recommendation_id)as num_recos
  
from "TASKRABBIT_PRODUCTION"."PUBLIC"."JOB_RECOMMENDATIONS" tr
join public.categories c
on tr.category_id = c.id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on tr.metro_id = m.metro_id
where tr.local_created_at >= '{start_date}'
and tr.last_in_funnel
and tr.locale = 'en-US'
group by 1,2,3,4,5
)

,
taskers as 
(
select count(distinct rca.rabbit_id) as num_taskers
  ,    TRUNC(rca.date,'day') as reco_day
  ,    rc.category_id as category_id
  ,    c.name as cat_name
from RABBIT_CALENDAR_AVAILABILITIES rca
join rabbit_profiles rp
on rca.rabbit_id = rp.user_id
join rabbit_categories rc
on rca.rabbit_id = rc.rabbit_id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on rp.geo_json_metro_id = m.metro_id
join public.categories c
on rc.category_id = c.id
where rca.date >= '{start_date}'
and rca.locale = 'en-US'
and rca.TOTAL_AVAILABILITY_THAT_DAY > 0
group by rca.date
  ,  rc.category_id
  ,  c.name
)
select recos.category_id
  ,    recos.cat_name
  ,    recos.reco_day as DAY
  ,    recos.reco_week as WEEK
  ,    recos.reco_month as MONTH
  ,    recos.num_recos
  ,    taskers.num_taskers
from recos
left join taskers
on (recos.category_id = taskers.category_id) and (recos.cat_name = taskers.cat_name) and (recos.reco_day = taskers.reco_day) 
""".format(start_date = start_date)

df_gb2 = df_supply.groupby(gb_cols, as_index = False)['number_of_results'].mean() 
df_gb2 = df_gb2.rename(columns = {'number_of_results':'avg_faces'})

#get price shown index
df_norm_px = sqlExec(query_norm_px)
df_norm_px = df_norm_px[~df_norm_px.price_shown_idx.isna()].reset_index(drop = True)
df_gb2 = df_gb2.merge(df_norm_px.groupby(gb_cols, as_index = False)['price_shown_idx'].mean())

#get reco per tasker
df_rpt_category = sqlExec(query_rpt_category)
df_rpt = (df_rpt_category.groupby(gb_cols).num_recos.sum() / df_rpt_category.groupby(gb_cols).num_taskers.sum()).reset_index()
df_rpt.columns = gb_cols + ['reco_per_tasker']

#make L2 df 
df_l2 = df_gb2.merge(df_rpt)
df_l2

#add zeros
df_zero = (df_supply[(df_supply['number_of_results'] == 0)].groupby(gb_cols).job_booked.count() / df_supply.groupby(gb_cols).job_booked.count()).reset_index()
df_zero = df_zero.rename(columns = {'job_booked':'zero_faces_rate'})

df_l2 = df_l2.merge(df_zero)

df = df_L1.merge(df_l2)


#l3s
#########################################
## pull utility data by metro          ##
## You will need to adjust this query  ##
##  if you want to group by categories ##
#########################################
query_ut_by_metro = """
with 
u as 
(select DISTINCT
  m.name as METRO_NAME, 
  TRUNC(rca.DATE,'day') as DAY,
  TRUNC(rca.DATE, 'week') as WEEK,
  TRUNC(rca.DATE, 'month') as MONTH,
  rca.rabbit_id,
  rca.total_availability_that_day as total_seconds,
  rca.appointment_seconds_booked as booked_seconds
from RABBIT_CALENDAR_AVAILABILITIES rca
left join rabbit_profiles rp
on rca.rabbit_id = rp.user_id
left join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on rp.geo_json_metro_id = m.metro_id
where rca.date >= '{start_date}'
and total_availability_that_day > 0
and rca.locale = 'en-US'
)

select 
METRO_NAME
,DAY
,WEEK
,MONTH
,SUM(TOTAL_SECONDS) as total_seconds
,SUM(BOOKED_SECONDS) as booked_seconds
,SUM(BOOKED_SECONDS) / SUM(TOTAL_SECONDS) as UTILIZATION
,AVG(TOTAL_SECONDS) as avg_total_seconds
,AVG(BOOKED_SECONDS) as avg_booked_seconds
,MEDIAN(TOTAL_SECONDS) as med_total_seconds
,MEDIAN(BOOKED_SECONDS) as med_booked_seconds
from u
group by 1,2,3,4
""".format(start_date = start_date)
#get wt price shown AND wt category experience
df_gb3 = df_norm_px.groupby(gb_cols, as_index = False)['wt_price_shown','wt_cat_experience'].mean()

#get zero face rate
df_zero = (df_supply[(df_supply['number_of_results'] == 0)].groupby(gb_cols).job_booked.count() / df_supply.groupby(gb_cols).job_booked.count()).reset_index()
df_zero = df_zero.rename(columns = {'job_booked':'zero_faces_rate'})
df_gb3 = df_gb3.merge(df_zero)


#df

#get hourly utilization AND avg_available_hours
df_metro_ut = run_sql_query(query_ut_by_metro)
df_metro_ut['HOURLY_UTILIZATION'] = df_metro_ut.UTILIZATION.astype(np.float64)
df_metro_ut.AVG_TOTAL_SECONDS = df_metro_ut.AVG_TOTAL_SECONDS.astype(np.float64)
df_metro_ut['AVG_AVAILABLE_HOURS'] = df_metro_ut.AVG_TOTAL_SECONDS / 3600
dfu = df_metro_ut.groupby(gb_cols, as_index = False)[['HOURLY_UTILIZATION','AVG_AVAILABLE_HOURS']].mean()

#make L3 df
df_L3 = df_gb3.merge(dfu)
df_L3

###########################
#sf specific
#
#
###########################

query_supply = """
select
distinct
jr.recommendation_id,
date_trunc('day', jr.created_at) as day,
date_trunc('week', jr.created_at) as week,
date_trunc('month', jr.created_at) as month,
jr.locale,
metros.name  as metro_name,
case when lower(categories.name) like '%ikea%' then 'IKEA' else categories.name end as cat_name,
case when jobs.id is NULL then 0  else 1 end as job_booked,
case when jr.number_of_results > 30 then 30 else jr.number_of_results end as number_of_results

from job_recommendations jr
LEFT JOIN jobs on jr.recommendation_id = jobs.recommendation_id
LEFT JOIN categories  ON jr.category_id  = categories.id
LEFT JOIN geographies  AS metros ON COALESCE(jr.metro_id, jr.geography_id)  = metros.id

WHERE jr.created_at  >= '{start_date}'
        and UPPER(jr.locale ) = UPPER('en-US')
        and jr.last_in_funnel 
        and (jobs.suppress_level < 1000 or jobs.suppress_level is NULL) 
        and jr.metro_id = 1053
""".format(start_date = start_date)

df_supply = sqlExec(query_supply)

gb_cols = ['month','cat_name']

#get recos
df_gb = df_supply.groupby(gb_cols, as_index = False).job_booked.count()
df_gb = df_gb.rename(columns = {'job_booked':'recos'})

#get jobs
df_gb = df_gb.merge(df_supply.groupby(gb_cols, as_index = False).job_booked.sum())
df_gb = df_gb.rename(columns = {'job_booked':'jobs'})

#get reco_conversion and market score
df_gb['reco_conversion'] = df_gb.jobs / df_gb.recos
df_gb['market_score'] = df_gb.reco_conversion * np.sqrt(df_gb.recos)

#make l1 df
df_L1 = df_gb[gb_cols + ['recos','reco_conversion','market_score']]
df_L1


#L2s
## Pull price information
query_norm_px = """with norm_px as(
select 
    DISTINCT
    job_recommendations.created_at as date,
    date_trunc('week', job_recommendations.created_at) as week,
    date_trunc('month', job_recommendations.created_at) as month,
    date_trunc('day', job_recommendations.created_at) as day,
    jrr.recommendation_id,
    LEAST(job_recommendations.NUMBER_OF_RESULTS,21) as MAX_POSITION,
    case when jobs.id is NULL then 0  else 1 end as job_booked,
    case when invitations.rabbit_id is NULL then 0 else 1 end as rabbit_booked,
    metros.name  as metro_name,
    case when lower(categories.name) like '%ikea%' then 'IKEA' else categories.name end as cat_name,
    case when job_recommendations.vehicle_requirement = 'truck' then 'truck'
         when job_recommendations.vehicle_requirement = 'car_or_truck' then 'car_or_truck'
         else 'none' end as vehicle_requirement,
    jrr.position,
    to_double(jrr.rabbit_hourly_rate_cents / 100.0) as rabbit_rate,
    ep.category_invoice_count,
    ep.expected_px,
    (to_double(jrr.rabbit_hourly_rate_cents / 100.0) - ep.expected_px)/ep.expected_px as px_error,
    100 * (1+ ((to_double(jrr.rabbit_hourly_rate_cents / 100.0) - ep.expected_px)/ep.expected_px)) as normalized_price    
    
FROM job_recommendation_recommendations jrr 
LEFT JOIN job_recommendations ON job_recommendations.recommendation_id = jrr.recommendation_id
LEFT JOIN categories  ON job_recommendations.category_id  = categories.id
LEFT JOIN geographies  AS metros ON COALESCE(job_recommendations.metro_id, job_recommendations.geography_id)  = metros.id
LEFT JOIN invitations ON jrr.recommendation_id = invitations.recommendation_id AND jrr.user_id = invitations.rabbit_id
LEFT JOIN jobs on jrr.recommendation_id = jobs.recommendation_id
left join playground.bb_expected_price ep on ep.METRO_NAME = metros.name and ep.CAT_NAME = (case when lower(categories.name) like '%ikea%' then 'IKEA' else categories.name end)
    and (case when job_recommendations.vehicle_requirement = 'truck' then 'truck'
         when job_recommendations.vehicle_requirement = 'car_or_truck' then 'car_or_truck'
         else 'none' end ) = (case when ep.vehicle_requirement is NULL then 'none' else ep.vehicle_requirement end)
    and (case when jrr.category_invoice_count > 75 then 75 else jrr.category_invoice_count end) = ep.category_invoice_count

WHERE job_recommendations.created_at  >= '{start_date}' and
      job_recommendations.metro_id = 1053 and
         (jrr.position  < 21) AND 
         ((UPPER(job_recommendations.locale ) = UPPER('en-US'))) AND 
          job_recommendations.last_in_funnel and
          (jobs.suppress_level < 1000 or jobs.suppress_level is NULL) 

order by recommendation_id, Position
)

select
norm_px.RECOMMENDATION_ID,
norm_px.date,
norm_px.day,
norm_px.week,
norm_px.month,
norm_px.METRO_NAME,
norm_px.CAT_NAME,
norm_px.VEHICLE_REQUIREMENT,
norm_px.MAX_POSITION,
sum(norm_px.normalized_price * w.WEIGHT) as price_shown_idx,
sum(norm_px.rabbit_rate * w.WEIGHT) as wt_price_shown,
sum(norm_px.category_invoice_count * w.WEIGHT) as wt_cat_experience,
max(norm_px.job_booked) as JOB_BOOKED
from

norm_px 
left join playground.bb_pos_weights w on w.MAX_POSITION = norm_px.MAX_POSITION and w.POSITION = norm_px.POSITION

group by 1,2,3,4,5,6,7,8,9
order by 2
""".format(start_date = start_date)

#########################################
## pull recos/taskers by metro         ##
## You will need to adjust this query  ##
##  if you want to group by categories ##
#########################################

query_rpt_metro = """
with recos as
(
select tr.metro_id
  ,    m.name as metro_name
  ,    TRUNC(tr.local_created_at,'day') reco_day
  ,    TRUNC(tr.local_created_at,'month') reco_month
  ,    TRUNC(tr.local_created_at,'week') reco_week
  ,    COUNT(DISTINCT tr.recommendation_id)as num_recos
  
from "TASKRABBIT_PRODUCTION"."PUBLIC"."JOB_RECOMMENDATIONS" tr
join public.categories c
on tr.category_id = c.id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on tr.metro_id = m.metro_id
where tr.local_created_at >= '{start_date}'
and tr.last_in_funnel
and tr.locale = 'en-US'
group by 1,2,3,4,5
)

,
taskers as 
(
select count(distinct rca.rabbit_id) as num_taskers
  ,    TRUNC(rca.date,'day') as reco_day
  ,    rp.geo_json_metro_id as metro_id
  ,    m.name as metro_name
from RABBIT_CALENDAR_AVAILABILITIES rca
join rabbit_profiles rp
on rca.rabbit_id = rp.user_id
join rabbit_categories rc
on rca.rabbit_id = rc.rabbit_id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on rp.geo_json_metro_id = m.metro_id
where rca.date >= '{start_date}'
and rca.locale = 'en-US'
and rca.TOTAL_AVAILABILITY_THAT_DAY > 0
group by rca.date
  ,  rp.geo_json_metro_id
  ,  m.name
)
select recos.metro_id
  ,    recos.metro_name
  ,    recos.reco_day as DAY
  ,    recos.reco_week as WEEK
  ,    recos.reco_month as MONTH
  ,    recos.num_recos
  ,    taskers.num_taskers
from recos
left join taskers
on (recos.metro_id = taskers.metro_id) and (recos.metro_name = taskers.metro_name) and (recos.reco_day = taskers.reco_day) 
""".format(start_date = start_date)

query_rpt_category = """
with recos as
(
select tr.category_id
  ,    c.name as cat_name
  ,    TRUNC(tr.local_created_at,'day') reco_day
  ,    TRUNC(tr.local_created_at,'month') reco_month
  ,    TRUNC(tr.local_created_at,'week') reco_week
  ,    COUNT(DISTINCT tr.recommendation_id)as num_recos
  
from "TASKRABBIT_PRODUCTION"."PUBLIC"."JOB_RECOMMENDATIONS" tr
join public.categories c
on tr.category_id = c.id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on tr.metro_id = m.metro_id
where tr.local_created_at >= '{start_date}'
and tr.last_in_funnel
and tr.locale = 'en-US'
and tr.metro_id = 1053
group by 1,2,3,4,5
)

,
taskers as 
(
select count(distinct rca.rabbit_id) as num_taskers
  ,    TRUNC(rca.date,'day') as reco_day
  ,    rc.category_id as category_id
  ,    c.name as cat_name
from RABBIT_CALENDAR_AVAILABILITIES rca
join rabbit_profiles rp
on rca.rabbit_id = rp.user_id
join rabbit_categories rc
on rca.rabbit_id = rc.rabbit_id
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on rp.geo_json_metro_id = m.metro_id
join public.categories c
on rc.category_id = c.id
where rca.date >= '{start_date}'
and rca.locale = 'en-US'
and rca.TOTAL_AVAILABILITY_THAT_DAY > 0
and rp.geo_json_metro_id = 1053
group by rca.date
  ,  rc.category_id
  ,  c.name
)
select recos.category_id
  ,    recos.cat_name
  ,    recos.reco_day as DAY
  ,    recos.reco_week as WEEK
  ,    recos.reco_month as MONTH
  ,    recos.num_recos
  ,    taskers.num_taskers
from recos
left join taskers
on (recos.category_id = taskers.category_id) and (recos.cat_name = taskers.cat_name) and (recos.reco_day = taskers.reco_day) 
""".format(start_date = start_date)

df_gb2 = df_supply.groupby(gb_cols, as_index = False)['number_of_results'].mean() 
df_gb2 = df_gb2.rename(columns = {'number_of_results':'avg_faces'})

#get price shown index
df_norm_px = sqlExec(query_norm_px)
df_norm_px = df_norm_px[~df_norm_px.price_shown_idx.isna()].reset_index(drop = True)
df_gb2 = df_gb2.merge(df_norm_px.groupby(gb_cols, as_index = False)['price_shown_idx'].mean())

#get reco per tasker
df_rpt_category = sqlExec(query_rpt_category)
df_rpt = (df_rpt_category.groupby(gb_cols).num_recos.sum() / df_rpt_category.groupby(gb_cols).num_taskers.sum()).reset_index()
df_rpt.columns = gb_cols + ['reco_per_tasker']

#make L2 df 
df_l2 = df_gb2.merge(df_rpt)
df_l2

#add zeros
df_zero = (df_supply[(df_supply['number_of_results'] == 0)].groupby(gb_cols).job_booked.count() / df_supply.groupby(gb_cols).job_booked.count()).reset_index()
df_zero = df_zero.rename(columns = {'job_booked':'zero_faces_rate'})

df_l2 = df_l2.merge(df_zero)

df = df_L1.merge(df_l2)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:49:07 2022

@author: wilhelmleinemann
"""

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import datetime as dt
from sklearn import preprocessing
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import folium


import snowflake.connector
from snowflake.connector import DictCursor
import json
import matplotlib.pyplot as plt

with open("snowflake_credentials.txt") as file:
    cred = json.load(file)
    
db_con = snowflake.connector.connect(**cred)
cur = db_con.cursor(DictCursor)

def sqlExec(query):
    df = pd.DataFrame(cur.execute(query).fetchall())
    df.columns = map(str.lower, df.columns)
    return df


sql = """
with ly as
(
SELECT
    metros.name  AS metro_name,
--    IFF ( categories.name = 'Help Moving', 'Moving Help', categories.name ) AS category_name,
    metros.id  AS metro_id,
--    2021 as year,
    job_recommendations.locale as locale,
    COUNT(DISTINCT job_recommendations.id ) AS num_recos,
    AVG(job_recommendations.number_of_results) as num_faces,
    COUNT(DISTINCT jobs.id ) AS num_jobs,
    COUNT(DISTINCT CASE WHEN job_invoices.approved_state_at  IS NOT NULL THEN job_invoices.id  ELSE NULL END) AS num_invoices,
    div0(sum(case when (job_invoices.first_poster_submitted_invoice = TRUE and job_invoices.approved_state_at is not null) then 1 else 0 end),
            sum(case when job_invoices.approved_state_at is not null then 1 else 0 end)) as pct_new_poster,
    COALESCE(CAST( ( SUM(DISTINCT (CAST(FLOOR(COALESCE( ( job_invoices.revenue_cents / 100.0  )  ,0)*(1000000*1.0)) AS DECIMAL(38,0))) + (TO_NUMBER(MD5( job_invoices.id  ), 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') % 1.0e27)::NUMERIC(38, 0) ) - SUM(DISTINCT (TO_NUMBER(MD5( job_invoices.id  ), 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') % 1.0e27)::NUMERIC(38, 0)) )  AS DOUBLE PRECISION) / CAST((1000000*1.0) AS DOUBLE PRECISION), 0) AS net_revenue
--    COUNT(DISTINCT job_invoices.id ) AS "job_invoices.count"
FROM job_recommendations                                       
LEFT JOIN jobs  AS jobs ON jobs.recommendation_id = job_recommendations.recommendation_id
LEFT JOIN job_invoices  AS job_invoices ON job_invoices.job_id= jobs.id
LEFT JOIN categories  AS categories ON job_recommendations.category_id  = categories.id
LEFT JOIN geographies  AS metros ON COALESCE(job_recommendations.metro_id, job_recommendations.geography_id)  = metros.id
WHERE ((( convert_timezone( 'GMT', (COALESCE( job_recommendations.tz_time_zone, 'UTC' )), job_recommendations.created_at )  ) >= (TO_TIMESTAMP('2020-02-01')) AND ( convert_timezone( 'GMT', (COALESCE( job_recommendations.tz_time_zone, 'UTC' )), job_recommendations.created_at )  ) < (TO_TIMESTAMP('2021-02-01')))) AND (job_recommendations.last_in_funnel ) AND ((jobs.suppress_level ) < 1000 OR (jobs.suppress_level ) IS NULL) AND (job_recommendations.locale is NOT NULL)
GROUP BY
    1,
    2,
    3
--    4
HAVING COUNT(DISTINCT job_recommendations.id ) > 0
ORDER BY
    4 DESC
)
,
ty as 
(
SELECT
    metros.name  AS metro_name,
--    IFF ( categories.name = 'Help Moving', 'Moving Help', categories.name ) AS category_name,
    metros.id  AS metro_id,
--    2022 as year,
    job_recommendations.locale as locale,    
    COUNT(DISTINCT job_recommendations.id ) AS num_recos,
    AVG(job_recommendations.number_of_results) as num_faces,
    COUNT(DISTINCT jobs.id ) AS num_jobs,
    COUNT(DISTINCT CASE WHEN job_invoices.approved_state_at  IS NOT NULL THEN job_invoices.id  ELSE NULL END) AS num_invoices,
    div0(sum(case when (job_invoices.first_poster_submitted_invoice = TRUE and job_invoices.approved_state_at is not null) then 1 else 0 end),
            sum(case when job_invoices.approved_state_at is not null then 1 else 0 end)) as pct_new_poster,
    COALESCE(CAST( ( SUM(DISTINCT (CAST(FLOOR(COALESCE( ( job_invoices.revenue_cents / 100.0  )  ,0)*(1000000*1.0)) AS DECIMAL(38,0))) + (TO_NUMBER(MD5( job_invoices.id  ), 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') % 1.0e27)::NUMERIC(38, 0) ) - SUM(DISTINCT (TO_NUMBER(MD5( job_invoices.id  ), 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') % 1.0e27)::NUMERIC(38, 0)) )  AS DOUBLE PRECISION) / CAST((1000000*1.0) AS DOUBLE PRECISION), 0) AS net_revenue
--    COUNT(DISTINCT job_invoices.id ) AS "job_invoices.count"
FROM job_recommendations
LEFT JOIN jobs  AS jobs ON jobs.recommendation_id = job_recommendations.recommendation_id
LEFT JOIN job_invoices  AS job_invoices ON job_invoices.job_id= jobs.id
LEFT JOIN categories  AS categories ON job_recommendations.category_id  = categories.id
LEFT JOIN geographies  AS metros ON COALESCE(job_recommendations.metro_id, job_recommendations.geography_id)  = metros.id
WHERE ((( convert_timezone( 'GMT', (COALESCE( job_recommendations.tz_time_zone, 'UTC' )), job_recommendations.created_at )  ) >= (TO_TIMESTAMP('2021-02-01')) AND ( convert_timezone( 'GMT', (COALESCE( job_recommendations.tz_time_zone, 'UTC' )), job_recommendations.created_at )  ) < (TO_TIMESTAMP('2022-02-01')))) AND (job_recommendations.last_in_funnel ) AND ((jobs.suppress_level ) < 1000 OR (jobs.suppress_level ) IS NULL)  AND (job_recommendations.locale is NOT NULL)
GROUP BY
    1,
    2,
    3
--    4
HAVING COUNT(DISTINCT job_recommendations.id ) > 0
ORDER BY
    4 DESC

)
,
cte_start_date as (
with tmp as (
select count(distinct j.ID) as num_jobs
  ,    DATE_TRUNC('month', j."CREATED_AT" ) as month
  ,    j.metro_id as metro_id
  ,    m.name as metro_name
from public.jobs j
join (select g.metro_id
        ,    g.name
        ,    g.time_zone
      from public.geographies g
      where g.id = g.metro_id) m
on j.metro_id = m.metro_id
--and   j.locale not in  ('en-US','en-CA', 'fr-CA')
group by  DATE_TRUNC('month', j."CREATED_AT" )
  ,    j.metro_id
  ,    m.name
order by month asc
)
select DISTINCT metro_id
  ,    DATEDIFF('month',first_value(month) over (partition by metro_id order by month nulls last),CURRENT_TIMESTAMP()) as months_active
from tmp
where num_jobs > 5
order by months_active
)

select ty.metro_name
--  ,    ty.category_name
  ,    ty.metro_id
  ,    ty.locale
--  ,    ty.num_recos as num_recos
--  ,    ly.num_recos as num_recos_ly
--  ,    ty.num_jobs as num_jobs
--  ,    ly.num_jobs as num_jobs_ly
  ,    ty.num_invoices as num_invoices
  ,    ly.num_invoices as num_invoices_ly
  ,    ty.net_revenue as net_revenue
  ,    ly.net_revenue as net_revenue_ly
  ,    ty.num_faces as num_faces
  ,    ly.num_faces as num_faces_ly
  ,    ty.pct_new_poster as pct_new_poster
--  ,    ly.pct_new_poster as pct_new_poster_ly
  ,    DIV0(ty.net_revenue,ty.num_recos) as rev_per_reco
--  ,    DIV0(ly.net_revenue,ly.num_recos) as rev_per_reco_ly
  ,    DIV0(ty.num_invoices,ty.num_recos) as invoice_ratio
--  ,    DIV0(ly.num_invoices,ly.num_recos) as invoice_ratio_ly
--  ,    DIV0(ty.num_recos,ly.num_recos) * 100 as reco_index
--  ,    DIV0(ty.num_jobs,ly.num_jobs) * 100 as job_index
  ,    DIV0(ty.num_invoices,ly.num_invoices) * 100 as invoice_index
  ,    DIV0(ty.net_revenue,ly.net_revenue) * 100 as net_revenue_index
  ,    DIV0(ty.num_faces,ly.num_faces) * 100 as faces_index
--  ,    DIV0(DIV0(ty.net_revenue,ty.num_recos),DIV0(ly.net_revenue,ly.num_recos)) * 100 as rpr_index
--  ,    DIV0(DIV0(ty.num_invoices,ty.num_recos),DIV0(ly.num_invoices,ly.num_recos)) * 100 as invoice_ratio_index
  ,    sd.months_active
from ty
left join ly
on (ty.metro_name = ly.metro_name 
    --and ty.category_name = ly.category_name 
    and ty.metro_id = ly.metro_id
    and ty.locale = ly.locale)
left join cte_start_date sd
on sd.metro_id = ty.metro_id
order by ty.num_recos desc


"""

df = sqlExec(sql)

def score_kmeans(df,n_clusters,weights,columns_to_drop):
    columns_to_drop.append(weights)
    kmeans = KMeans(n_clusters = n_clusters, random_state=0, max_iter=1000)
    X = np.array(df.drop(columns_to_drop,1).astype(float))
    Y = np.array(df[weights].astype(float))
    
    wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
    predicted_kmeans = kmeans.predict(X,sample_weight=Y)
    kmeans_score = kmeans.score(X,sample_weight = Y)
    
    return wt_kmeans_clusters, predicted_kmeans, kmeans_score



#kmeans cluster tuning
scores = []
K_clusters = range(1,10,1)

for i in K_clusters:
    wkc, pk, sc = score_kmeans(df,5,'num_invoices',['metro_id','metro_name'])
    scores.append((i,sc))


number_of_clusters = 6

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df.drop(['metro_id','metro_name'],1).astype(float))
Y = np.array(df['num_invoices'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight=Y)
kmeans_score = kmeans.score(X,sample_weight = Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('SF Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df['kmeans_cluster'] = predicted_kmeans

df_sf_g.groupby('kmeans_cluster').sum()

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['kmeans_cluster'] = df_centers.index

quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                         n_quantiles = 20,
                                                         random_state=0)

robust_scaler = preprocessing.RobustScaler()
df = df.fillna(0)
df_features = df.drop(['metro_name','metro_id'],axis=1)
df_features = df[['num_invoices','num_faces','months_active','num_invoices_ly','rev_per_reco','invoice_ratio','pct_new_poster']]
df_features = df[['num_invoices','months_active','num_faces','num_invoices_ly']]
df_features = df[['num_invoices']]
df_features_qt = quantile_transformer.fit_transform(np.array(df_features))
df_features_rs = robust_scaler.fit_transform(np.array(df_features))

kmeans = KMeans(n_clusters = 7, random_state=0, max_iter=1000)
kmeans.fit(df_features_qt)
predicted_kmeans = kmeans.predict(df_features_qt)

kmeans.fit(df_features_rs)
predicted_kmeans = kmeans.predict(df_features_rs)


df_c = df.copy()
df_c['cluster'] = predicted_kmeans

df_c = df_c[['metro_name','metro_id','locale','num_invoices','num_faces','months_active','num_invoices_ly','rev_per_reco','invoice_ratio','pct_new_poster','cluster']]
df_c.to_csv('all_growth_metros.csv',index=False)

scores = []
K_clusters = range(1,15,1)

for i in K_clusters:
    kmeans = KMeans(n_clusters = i, random_state=0, max_iter=1000)
    kmeans.fit(df_features_rs)
    predicted_kmeans = kmeans.predict(df_features_rs)
    sc = kmeans.score(df_features_rs)
    
    scores.append((i,sc))

df_scores = pd.DataFrame(scores, columns=['number_of_clusters','score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])



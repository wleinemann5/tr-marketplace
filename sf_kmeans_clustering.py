#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:43:49 2022

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


lat_lng_sql = """
select ROUND(jr.lat,4) as lat
  ,    ROUND(jr.lng,4) as lng
  ,    count(distinct recommendation_id) as num_recos
from job_recommendations jr
where jr.metro_id in (1053)
and jr.created_at >= '2021-01-27'
and jr.created_at <= '2022-01-26'
and jr.last_in_funnel
and jr.postal_code is not null
and jr.lng < -121
group by ROUND(jr.lat,4)
  ,      ROUND(jr.lng,4)
order by num_recos desc
"""

df = sqlExec(lat_lng_sql)

plt.style.use('default')
x=np.array(df['lng'])
y=np.array(df['lat'])
plt.figure(figsize=(15, 10))
plt.scatter(x, y, s=5, cmap='viridis',c='orange',label='num_recos')
plt.title('SF Reco Locations',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)

number_of_clusters = 100

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df.drop(['num_recos'],1).astype(float))
Y = np.array(df['num_recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight = Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('SF Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df['cluster_id_wt'] = predicted_kmeans
df_g = df.groupby(by='cluster_id_wt',as_index=False).sum().sort_values(by='num_recos',ascending=False)


########
## just sf
########

sf_lat_lng_sql = """
select ROUND(jr.lat,4) as lat
  ,    ROUND(jr.lng,4) as lng
  ,    count(distinct recommendation_id) as num_recos
from job_recommendations jr
where jr.metro_id in (1053)
and jr.created_at >= '2021-01-27'
and jr.created_at <= '2022-01-26'
and jr.last_in_funnel
and jr.postal_code is not null
and jr.lng < -121
and jr.postal_code in (94109,94110,94103,94107,94115,94117,94114,94123,94118,94105,94102,94122,94131,94121,94133,94116,94112,94158,94111,94124,94108,94127,94132,94134,94104,94129,94128,94130)
group by ROUND(jr.lat,4)
  ,      ROUND(jr.lng,4)
order by num_recos desc
"""

df = sqlExec(sf_lat_lng_sql)

plt.style.use('default')
x=np.array(df['lng'])
y=np.array(df['lat'])
plt.figure(figsize=(15, 10))
plt.scatter(x, y, s=5, cmap='viridis',c='orange',label='num_recos')
plt.title('SF Reco Locations',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)


K_clusters = range(1,25)

scores = []
for i in K_clusters:

    kmeans = KMeans(n_clusters = i, random_state=0, max_iter=1000)
    X = np.array(df.drop(['num_recos'],1).astype(float))
    Y = np.array(df['num_recos'].astype(float))

    wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
    predicted_kmeans = kmeans.predict(X,sample_weight = Y)
    scores.append((i,kmeans.score(X,sample_weight=Y)))

df_scores = pd.DataFrame(scores,columns=['number_of_clusters','score'])
df_scores['log_score'] = np.log(np.abs(df_scores['score']))
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 9

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df.drop(['num_recos'],1).astype(float))
Y = np.array(df['num_recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight = Y)


plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('SF Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['cluster_number'] = df_centers.index

df['cluster_id_wt'] = predicted_kmeans
df_g = df.groupby(by='cluster_id_wt',as_index=False).sum().sort_values(by='num_recos',ascending=False)

df_centers = df_centers.merge(df_g,how='left',left_on='cluster_number',right_on='cluster_id_wt')
df_centers_sf = df_centers

map_sf = folium.Map(location=[df.lat.mean(),df.lng.mean()],zoom_start=14,control_scale=(True))

for index, location_info in df_centers.iterrows():
    folium.Marker([location_info["lat_center"], location_info["lng_center"]], popup=location_info["num_recos"]).add_to(map_sf)

map_sf.save('sf_map.html')


######
# just oakland
######

oak_lat_lng_sql = """
select ROUND(jr.lat,4) as lat
  ,    ROUND(jr.lng,4) as lng
  ,    count(distinct recommendation_id) as num_recos
from job_recommendations jr
where jr.metro_id in (1053)
and jr.created_at >= '2021-01-27'
and jr.created_at <= '2022-01-26'
and jr.last_in_funnel
and jr.postal_code is not null
and jr.lng < -121
and jr.postal_code in (94608,94607,94501,94609,94618,94611,94602,94610,94606,94612)
group by ROUND(jr.lat,4)
  ,      ROUND(jr.lng,4)
order by num_recos desc
"""

df = sqlExec(oak_lat_lng_sql)

plt.style.use('default')
x=np.array(df['lng'])
y=np.array(df['lat'])
plt.figure(figsize=(15, 10))
plt.scatter(x, y, s=5, cmap='viridis',c='orange',label='num_recos')
plt.title('SF Reco Locations',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)


K_clusters = range(1,20)

scores = []
for i in K_clusters:

    kmeans = KMeans(n_clusters = i, random_state=0, max_iter=1000)
    X = np.array(df.drop(['num_recos'],1).astype(float))
    Y = np.array(df['num_recos'].astype(float))

    wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
    predicted_kmeans = kmeans.predict(X,sample_weight = Y)
    scores.append((i,kmeans.score(X,sample_weight=Y)))

df_scores = pd.DataFrame(scores,columns=['number_of_clusters','score'])
df_scores['log_score'] = np.log(np.abs(df_scores['score']))
#plt.scatter(df_scores['number_of_clusters'],df_scores['log_score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 7

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df.drop(['num_recos'],1).astype(float))
Y = np.array(df['num_recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight = Y)
kmeans.score(X,sample_weight=Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('SF Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['cluster_number'] = df_centers.index

df['cluster_id_wt'] = predicted_kmeans
df_g = df.groupby(by='cluster_id_wt',as_index=False).sum().sort_values(by='num_recos',ascending=False)

df_centers = df_centers.merge(df_g,how='left',left_on='cluster_number',right_on='cluster_id_wt')
df_centers_oak = df_centers

map_oak = folium.Map(location=[df.lat.mean(),df.lng.mean()],zoom_start=14,control_scale=(True))

for index, location_info in df_centers.iterrows():
    folium.Marker([location_info["lat_center"], location_info["lng_center"]], popup=location_info["num_recos"]).add_to(map_oak)

map_oak.save('oak_map.html')


#######
#outside of sf & oak
#######

no_sf_oak_lat_lng_sql = """
select ROUND(jr.lat,4) as lat
  ,    ROUND(jr.lng,4) as lng
  ,    count(distinct recommendation_id) as num_recos
from job_recommendations jr
where jr.metro_id in (1053)
and jr.created_at >= '2021-01-27'
and jr.created_at <= '2022-01-26'
and jr.last_in_funnel
and jr.postal_code is not null
and jr.lng < -121
and jr.postal_code not in (94109,94110,94103,94107,94115,94117,94114,94123,94118,94105,94102,94122,94131,94121,94133,94116,94112,94158,94111,94124,94108,94127,94132,94134,94104,94129,94128,94130,94608,94607,94501,94609,94618,94611,94602,94610,94606,94612)
group by ROUND(jr.lat,4)
  ,      ROUND(jr.lng,4)
order by num_recos desc
"""

df = sqlExec(no_sf_oak_lat_lng_sql)

plt.style.use('default')
x=np.array(df['lng'])
y=np.array(df['lat'])
plt.figure(figsize=(15, 10))
plt.scatter(x, y, s=5, cmap='viridis',c='orange',label='num_recos')
plt.title('SF Reco Locations',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)


K_clusters = range(1,100,5)

scores = []
for i in K_clusters:

    kmeans = KMeans(n_clusters = i, random_state=0, max_iter=1000)
    X = np.array(df.drop(['num_recos'],1).astype(float))
    Y = np.array(df['num_recos'].astype(float))

    wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
    predicted_kmeans = kmeans.predict(X,sample_weight = Y)
    scores.append((i,kmeans.score(X,sample_weight=Y)))

df_scores = pd.DataFrame(scores,columns=['number_of_clusters','score'])
df_scores['log_score'] = np.log(np.abs(df_scores['score']))
#plt.scatter(df_scores['number_of_clusters'],df_scores['log_score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 40

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df.drop(['num_recos'],1).astype(float))
Y = np.array(df['num_recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight = Y)
kmeans.score(X,sample_weight=Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('SF Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['cluster_number'] = df_centers.index

df['cluster_id_wt'] = predicted_kmeans
df_g = df.groupby(by='cluster_id_wt',as_index=False).sum().sort_values(by='num_recos',ascending=False)

df_centers = df_centers.merge(df_g,how='left',left_on='cluster_number',right_on='cluster_id_wt')

df_centers_bay = pd.concat([df_centers,df_centers_sf,df_centers_oak],ignore_index=True)
map_bay_area = folium.Map(location=[df.lat.mean(),df.lng.mean()],zoom_start=10,control_scale=(True))

for index, location_info in df_centers_bay.iterrows():
    folium.Marker([location_info["lat_center"], location_info["lng_center"]], popup=location_info["num_recos"]).add_to(map_bay_area)

map_bay_area.save('bay_area_map.html')

#h3

import h3
df.head()
#h3.geo_to_h3(df['lat'][0],df['lng'][0],df['resolution'][0])

def plot_scatter(df, metric_col, x='lng', y='lat', marker='.', alpha=1, figsize=(16,12), colormap='viridis'):    
    df.plot.scatter(x=x, y=y, c=metric_col, title=metric_col
                    , edgecolors='none', colormap=colormap, marker=marker, alpha=alpha, figsize=figsize);
    plt.xticks([], []); plt.yticks([], [])
    
def aperture_downsampling(df, hex_col, metric_col, coarse_aperture_size):
    df_coarse = df.copy()
    coarse_hex_col = 'hex{}'.format(coarse_aperture_size)
    df_coarse[coarse_hex_col] = df_coarse[hex_col].apply(lambda x: h3.h3_to_parent(x,coarse_aperture_size))
    dfc = df_coarse.groupby([coarse_hex_col])[[metric_col,]].mean().reset_index()
    dfc['lat'] = dfc[coarse_hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    dfc['lng'] = dfc[coarse_hex_col].apply(lambda x: h3.h3_to_geo(x)[1]) 
    return dfc

def kring_smoothing(df, hex_col, metric_col, k):
    dfk = df[[hex_col]] 
    dfk.index = dfk[hex_col]
    dfs =  (dfk[hex_col]
                 .apply(lambda x: pd.Series(list(h3.k_ring(x,k)))).stack()
                 .to_frame('hexk').reset_index(1, drop=True).reset_index()
                 .merge(df[[hex_col,metric_col]]).fillna(0)
                 .groupby(['hexk'])[[metric_col]].sum().divide((1 + 3 * k * (k + 1)))
                 .reset_index()
                 .rename(index=str, columns={"hexk": hex_col}))
    dfs['lat'] = dfs[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    dfs['lng'] = dfs[hex_col].apply(lambda x: h3.h3_to_geo(x)[1]) 
    return dfs

#for sf
resolution = 10
df['resolution'] = resolution

#find hexes
df['hex_col'] = df.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,x.resolution),1)

#aggregate the recos
df_hex_g = df.groupby('hex_col').sum()['num_recos'].to_frame('recos').reset_index()

#find center of hex for visualization
df_hex_g['lat'] = df_hex_g['hex_col'].apply(lambda x: h3.h3_to_geo(x)[0])
df_hex_g['lng'] = df_hex_g['hex_col'].apply(lambda x: h3.h3_to_geo(x)[1])

#plot hexes
plot_scatter(df_hex_g,metric_col='recos',marker='o',figsize=(17,15))

#kring_smoothing
k = 12
df_hex_s = kring_smoothing(df_hex_g,'hex_col',metric_col='recos',k=k)
print('sum sanity check:', df_hex_s['recos'].sum() / df_hex_g['recos'].sum())
plot_scatter(df_hex_s, metric_col='recos', marker='o',figsize=(17,15))
plt.title('bay area_recos: 4-ring average');



K_clusters = range(1,20)

scores = []
for i in K_clusters:

    kmeans = KMeans(n_clusters = i, random_state=0, max_iter=1000)
    X = np.array(df_hex_g.drop(['recos','hex_col'],1).astype(float))
    Y = np.array(df_hex_g['recos'].astype(float))

    wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
    predicted_kmeans = kmeans.predict(X,sample_weight = Y)
    scores.append((i,kmeans.score(X,sample_weight=Y)))

df_scores = pd.DataFrame(scores,columns=['number_of_clusters','score'])
df_scores['log_score'] = np.log(np.abs(df_scores['score']))
#plt.scatter(df_scores['number_of_clusters'],df_scores['log_score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 9

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df_hex_g.drop(['recos','hex_col'],1).astype(float))
Y = np.array(df_hex_g['recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight = Y)
kmeans.score(X,sample_weight=Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('SF Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['cluster_number'] = df_centers.index

df['cluster_id_wt'] = predicted_kmeans
df_g = df.groupby(by='cluster_id_wt',as_index=False).sum().sort_values(by='num_recos',ascending=False)

df_centers = df_centers.merge(df_g,how='left',left_on='cluster_number',right_on='cluster_id_wt')
df_centers_oak = df_centers

map_oak = folium.Map(location=[df.lat.mean(),df.lng.mean()],zoom_start=14,control_scale=(True))

for index, location_info in df_centers.iterrows():
    folium.Marker([location_info["lat_center"], location_info["lng_center"]], popup=location_info["num_recos"]).add_to(map_oak)

map_oak.save('oak_map.html')




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 00:34:42 2022

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
import h3

import snowflake.connector
from snowflake.connector import DictCursor
import json
import matplotlib.pyplot as plt
import matplotlib

import folium
from geojson import Feature, Point, FeatureCollection
import json

with open("snowflake_credentials.txt") as file:
    cred = json.load(file)
    
db_con = snowflake.connector.connect(**cred)
cur = db_con.cursor(DictCursor)

def sqlExec(query):
    df = pd.DataFrame(cur.execute(query).fetchall())
    df.columns = map(str.lower, df.columns)
    return df

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

def add_h3_data(df, hex_col, resolution):
    df_c = df.copy()
    df_c['resolution'] = resolution
    
    #find hexes
    df_c[hex_col] = df_c.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,x.resolution),1)
    
    #aggregate the recos
    df_hex_g = df_c.groupby(hex_col).sum()['num_recos'].to_frame('num_recos').reset_index()
    
    #find center of hex for visualization
    df_hex_g['lat'] = df_hex_g[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    df_hex_g['lng'] = df_hex_g[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
    
    return df_c, df_hex_g

def score_kmeans(df,n_clusters,weights,columns_to_drop):
    columns_to_drop.append(weights)
    kmeans = KMeans(n_clusters = n_clusters, random_state=0, max_iter=1000)
    X = np.array(df.drop(columns_to_drop,1).astype(float))
    Y = np.array(df[weights].astype(float))
    
    wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
    predicted_kmeans = kmeans.predict(X,sample_weight=Y)
    kmeans_score = kmeans.score(X,sample_weight = Y)
    
    return wt_kmeans_clusters, predicted_kmeans, kmeans_score


def hexagons_dataframe_to_geojson(df_hex, file_output = None, column_name = "value"):
    """
    Produce the GeoJSON for a dataframe, constructing the geometry from the "hex_id" column
    and with a property matching the one in column_name
    """    
    list_features = []
    
    for i,row in df_hex.iterrows():
        try:
            geometry_for_row = { "type" : "Polygon", "coordinates": [h3.h3_to_geo_boundary(h=row["hex_id"],geo_json=True)]}
            feature = Feature(geometry = geometry_for_row , id=row["hex_id"], properties = {column_name : row[column_name]})
            list_features.append(feature)
        except:
            print("An exception occurred for hex " + row["hex_id"]) 

    feat_collection = FeatureCollection(list_features)
    geojson_result = json.dumps(feat_collection)
    return geojson_result

def get_color(custom_cm, val, vmin, vmax):
    return matplotlib.colors.to_hex(custom_cm((val-vmin)/(vmax-vmin)))

def choropleth_map(df_aggreg, column_name = "value", border_color = 'black', fill_opacity = 0.7, color_map_name = "Blues", initial_map = None):
    """
    Creates choropleth maps given the aggregated data. initial_map can be an existing map to draw on top of.
    """    
    #colormap
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
    print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")
    
    # the name of the layer just needs to be unique, put something silly there for now:
    name_layer = "Choropleth " + str(df_aggreg)
    
    if initial_map is None:
        initial_map = folium.Map(location= [47, 4], zoom_start=5.5, tiles="cartodbpositron")

    #create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg, column_name = column_name)

    # color_map_name 'Blues' for now, many more at https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose from!
    custom_cm = matplotlib.cm.get_cmap(color_map_name)

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': get_color(custom_cm, feature['properties'][column_name], vmin=min_value, vmax=max_value),
            'color': border_color,
            'weight': 1,
            'fillOpacity': fill_opacity 
        }, 
        name = name_layer
    ).add_to(initial_map)

    return initial_map


#get sf bay area location and reco data
all_sf_sql = """
select ROUND(jr.lat,4) as lat
  ,    ROUND(jr.lng,4) as lng
  ,    jr.postal_code
  ,    pc.city
  ,    count(distinct recommendation_id) as num_recos
from job_recommendations jr
left join dw_postal_codes_world pc
on jr.postal_code = pc.postal_code
where jr.metro_id in (1053)
and jr.created_at >= '2021-01-27'
and jr.created_at <= '2022-01-26'
and jr.last_in_funnel
and jr.lng < -121
group by ROUND(jr.lat,4)
  ,      ROUND(jr.lng,4)
  ,      jr.postal_code
  ,      pc.city
order by num_recos desc
"""

df = sqlExec(all_sf_sql)

#investigate city densitys
df_g_city = df.groupby(by='city',as_index=False).sum()

#visually inspect cities with over 9k recos.
#are there locations that need higher resolution groupings?

#segment out sf, oakland area, san jose area
idx_sf = (df['city'] == 'San Francisco')
idx_oak = ((df['city'] == 'Oakland') | (df['city'] == 'Berkeley') | (df['city'] == 'Alameda') | (df['city'] == 'Emeryville'))
idx_sj = ((df['city'] == 'San Jose') | (df['city'] == 'Sunnyvale') | (df['city'] == 'Santa Clara') | (df['city'] == 'Campbell') | (df['city'] == 'Cupertino'))

df_sf = df.loc[idx_sf]
df_oak = df.loc[idx_oak]
df_sj = df.loc[idx_sj]
df_else = df.loc[~idx_sf].loc[~idx_oak].loc[~idx_sj]
df_else = df_else[~df_else.postal_code.isna()]
#######
#sf
#######

hex_col = 'hex_id'
df_sf, df_sf_g = add_h3_data(df_sf,hex_col,8)    

#plot hexes
plot_scatter(df_sf_g,metric_col='num_recos',marker='o',figsize=(17,15))
plt.title('sf_recos')

#kring_smoothing
k = 2
df_sf_s = kring_smoothing(df_sf_g,hex_col,metric_col='num_recos',k=k)
print('sum sanity check:', df_sf_s['num_recos'].sum() / df_sf_g['num_recos'].sum())
plot_scatter(df_sf_s, metric_col='num_recos', marker='o',figsize=(17,15))
plt.title('sf_recos: 2-ring average');


#kmeans cluster tuning
scores = []
K_clusters = range(1,20,1)

for i in K_clusters:
    wkc, pk, sc = score_kmeans(df_sf_g,i,'num_recos',[hex_col])
    scores.append((i,sc))

df_scores = pd.DataFrame(scores, columns=['number_of_clusters','score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 9

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df_sf_g.drop(['num_recos',hex_col],1).astype(float))
Y = np.array(df_sf_g['num_recos'].astype(float))

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

df_sf_g['kmeans_cluster'] = predicted_kmeans

df_sf_g.groupby('kmeans_cluster').sum()

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['kmeans_cluster'] = df_centers.index

df_centers = df_centers.merge(df_sf_g.groupby('kmeans_cluster',as_index=False).sum()[['kmeans_cluster','num_recos']],how='left',on='kmeans_cluster')

df_sf_g = df_sf_g.merge(df_centers,how='left',on='kmeans_cluster')

map_sf = folium.Map(location=[df_sf_g.lat.mean(),df_sf_g.lng.mean()],zoom_start=14,control_scale=(True))

for index, location_info in df_centers.iterrows():
    folium.Marker([location_info["lat_center"], location_info["lng_center"]], popup=location_info["num_recos"]).add_to(map_sf)

map_sf.save('sf_map.html')

map_sf = folium.Map(location=[df_sf_g.lat.mean(),df_sf_g.lng.mean()],zoom_start=10,control_scale=(True),tiles="cartodbpositron")

map_sf = choropleth_map(df_sf_g,'num_recos',initial_map=map_sf)

map_sf.save('sf_map.html')


df_sf_g.groupby('kmeans_cluster').sum()

###dont forget about Treasure Island

#####
#oak
#####

hex_col = 'hex_id'
df_oak, df_oak_g = add_h3_data(df_oak,hex_col,7)    

#plot hexes
plot_scatter(df_oak_g,metric_col='num_recos',marker='o',figsize=(17,15))
plt.title('oak_recos')

#kring_smoothing
k = 2
df_oak_s = kring_smoothing(df_oak_g,hex_col,metric_col='num_recos',k=k)
print('sum sanity check:', df_oak_s['num_recos'].sum() / df_oak_g['num_recos'].sum())
plot_scatter(df_oak_s, metric_col='num_recos', marker='o',figsize=(17,15))
plt.title('oak_recos: 2-ring average');

#kmeans cluster tuning
scores = []
K_clusters = range(1,20,1)

for i in K_clusters:
    wkc, pk, sc = score_kmeans(df_oak_g,i,'num_recos',[hex_col])
    scores.append((i,sc))

df_scores = pd.DataFrame(scores, columns=['number_of_clusters','score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 8

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df_oak_g.drop(['num_recos',hex_col],1).astype(float))
Y = np.array(df_oak_g['num_recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight=Y)
kmeans_score = kmeans.score(X,sample_weight = Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('OAK Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df_oak_g['kmeans_cluster'] = predicted_kmeans

df_oak_g.groupby('kmeans_cluster').sum()

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['kmeans_cluster'] = df_centers.index

df_centers = df_centers.merge(df_oak_g.groupby('kmeans_cluster',as_index=False).sum()[['kmeans_cluster','num_recos']],how='left',on='kmeans_cluster')

map_oak = folium.Map(location=[df_oak_g.lat.mean(),df_oak_g.lng.mean()],zoom_start=14,control_scale=(True))

for index, location_info in df_centers.iterrows():
    folium.Marker([location_info["lat_center"], location_info["lng_center"]], popup=location_info["num_recos"]).add_to(map_oak)

map_oak.save('oak_map.html')

map_oak = folium.Map(location=[df_oak_g.lat.mean(),df_oak_g.lng.mean()],zoom_start=10,control_scale=(True),tiles="cartodbpositron")

map_oak = choropleth_map(df_oak_g,'num_recos',initial_map=map_oak)

map_oak.save('oak_map.html')


#####
#sj
#####

hex_col = 'hex_id'
df_sj, df_sj_g = add_h3_data(df_sj,hex_col,7)    

#plot hexes
plot_scatter(df_sj_g,metric_col='num_recos',marker='o',figsize=(17,15))
plt.title('sj_recos')

#kring_smoothing
k = 2
df_sj_s = kring_smoothing(df_sj_g,hex_col,metric_col='num_recos',k=k)
print('sum sanity check:', df_sj_s['num_recos'].sum() / df_sj_g['num_recos'].sum())
plot_scatter(df_sj_s, metric_col='num_recos', marker='o',figsize=(17,15))
plt.title('oak_recos: 2-ring average');

#kmeans cluster tuning
scores = []
K_clusters = range(1,20,1)

for i in K_clusters:
    wkc, pk, sc = score_kmeans(df_sj_g,i,'num_recos',[hex_col])
    scores.append((i,sc))

df_scores = pd.DataFrame(scores, columns=['number_of_clusters','score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 7

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df_sj_g.drop(['num_recos',hex_col],1).astype(float))
Y = np.array(df_sj_g['num_recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight=Y)
kmeans_score = kmeans.score(X,sample_weight = Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('OAK Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df_sj_g['kmeans_cluster'] = predicted_kmeans

df_sj_g.groupby('kmeans_cluster').sum()

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['kmeans_cluster'] = df_centers.index

df_centers = df_centers.merge(df_sj_g.groupby('kmeans_cluster',as_index=False).sum()[['kmeans_cluster','num_recos']],how='left',on='kmeans_cluster')

map_sj = folium.Map(location=[df_sj_g.lat.mean(),df_sj_g.lng.mean()],zoom_start=10,control_scale=(True))

for index, location_info in df_centers.iterrows():
    folium.Marker([location_info["lat_center"], location_info["lng_center"]], popup=location_info["num_recos"]).add_to(map_sj)

map_sj.save('sj_map.html')

map_sj = folium.Map(location=[df_sj_g.lat.mean(),df_sj_g.lng.mean()],zoom_start=10,control_scale=(True),tiles="cartodbpositron")

map_sj = choropleth_map(df_sj_g,'num_recos',initial_map=map_sj)

map_sj.save('sj_map.html')


#####
#else
#####

hex_col = 'hex_id'
df_else, df_else_g = add_h3_data(df_else,hex_col,7)    

#plot hexes
plot_scatter(df_else_g,metric_col='num_recos',marker='o',figsize=(17,15))
plt.title('bay_area_recos')

#kring_smoothing
k = 2
df_else_s = kring_smoothing(df_else_g,hex_col,metric_col='num_recos',k=k)
print('sum sanity check:', df_else_s['num_recos'].sum() / df_else_g['num_recos'].sum())
plot_scatter(df_else_s, metric_col='num_recos', marker='o',figsize=(17,15))
plt.title('bay_area_recos: 2-ring average');

#kmeans cluster tuning
scores = []
K_clusters = range(1,30,5)

for i in K_clusters:
    wkc, pk, sc = score_kmeans(df_else_g,i,'num_recos',[hex_col])
    scores.append((i,sc))

df_scores = pd.DataFrame(scores, columns=['number_of_clusters','score'])
plt.scatter(df_scores['number_of_clusters'],df_scores['score'])

number_of_clusters = 35

kmeans = KMeans(n_clusters = number_of_clusters, random_state=0, max_iter=1000)
X = np.array(df_else_g.drop(['num_recos',hex_col],1).astype(float))
Y = np.array(df_else_g['num_recos'].astype(float))

wt_kmeans_clusters = kmeans.fit(X,sample_weight=Y)
predicted_kmeans = kmeans.predict(X,sample_weight=Y)
kmeans_score = kmeans.score(X,sample_weight = Y)

plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:,1], X[:,0], c=wt_kmeans_clusters.labels_.astype(float),s=10,cmap='tab20b',marker='x')
plt.title('Bay Area Reco Locations - Weighted K-Means',fontsize=18, fontweight='bold')
plt.xlabel('lng',fontsize=15)
plt.ylabel('lat',fontsize=15)
centers = wt_kmeans_clusters.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=500, alpha=0.5);

df_else_g['kmeans_cluster'] = predicted_kmeans

df_else_g.groupby('kmeans_cluster').sum()

df_centers = pd.DataFrame(centers,columns=['lat_center','lng_center'])
df_centers['kmeans_cluster'] = df_centers.index

df_centers = df_centers.merge(df_else_g.groupby('kmeans_cluster',as_index=False).sum()[['kmeans_cluster','num_recos']],how='left',on='kmeans_cluster')

map_else = folium.Map(location=[df_else_g.lat.mean(),df_else_g.lng.mean()],zoom_start=10,control_scale=(True),tiles="cartodbpositron")

map_else = choropleth_map(df_else_g,'num_recos',initial_map=map_else)

map_else.save('bay_area_map.html')




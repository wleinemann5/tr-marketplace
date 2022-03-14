#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:51:04 2022

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

def add_h3_data(df, hex_col, metric_col, resolution):
    df_c = df.copy()
    df_c['resolution'] = resolution
    
    #find hexes
    df_c[hex_col] = df_c.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,x.resolution),1)
    
    #aggregate the recos
    df_hex_g = df_c.groupby(hex_col).sum()[metric_col].to_frame(metric_col).reset_index()
    
    #find center of hex for visualization
    df_hex_g['lat'] = df_hex_g[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    df_hex_g['lng'] = df_hex_g[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
    
    df_hex_g['resolution'] = resolution
    
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
with cte_tt as
(
select jrr.user_id
  ,    jr.recommendation_id
  ,    least(jr.number_of_results,200) as number_of_results
  ,    least(jrr.position,200) as position
  ,    ji.id as job_invoice_id
from public.job_recommendations jr
join public.job_recommendation_recommendations jrr
on  jr.recommendation_id = jrr.recommendation_id
left join invitations i
on jr.recommendation_id = i.recommendation_id and jrr.user_id = i.rabbit_id
left join jobs j
on j.recommendation_id = jr.recommendation_id
left join job_invoices ji
on j.id = ji.job_id and jrr.user_id = ji.rabbit_id
where jr.local_created_at >= DATEADD('year',-2,CURRENT_DATE())
and jr.local_created_at <= DATEADD('day',-7,CURRENT_DATE())
and jr.last_in_funnel
and jr.locale not in ('en-US')
and j.fixup_key is NULL
and (j.suppress_level < 1000 or j.suppress_level is NULL)
and UPPER(ji.state) in ('MANUALLY_PAID','APPROVED')
)
,
cte_invoice_weights as
(
select number_of_results
  ,    position
  ,    DIV0(COUNT(DISTINCT job_invoice_id),SUM(COUNT(DISTINCT job_invoice_id)) over (partition by number_of_results)) as invoice_pct
from cte_tt
where position <= number_of_results
group by position
  ,      number_of_results
order by number_of_results asc
  ,      position asc
)

--pull in position weights to adjust for value in jrr table
--cte_reco_table as
--(
select jr.postal_code
  ,    jr.metro_id
  ,    jr.category_id
  ,    pc.city
  ,    ROUND(jr.lat,4) as lat
  ,    ROUND(jr.lng,4) as lng
  ,    count(distinct recommendation_id) as num_recos
  ,    SUM(iw.invoice_pct) as opportunity
from public.job_recommendations jr
left join dw_postal_codes_world pc
on jr.postal_code = pc.postal_code
left join cte_invoice_weights iw
on iw.position = CAST((LEAST(jr.number_of_results,150) + 1)/2 as integer) and iw.number_of_results = (LEAST(jr.number_of_results,150) + 1)
where jr.last_in_funnel
--and jr.metro_id in (1053)
and jr.local_created_at >= '2022-02-01'
and jr.local_created_at <= '2022-02-28'
and jr.locale in ('en-US')
--and jr.category_id in (113)
group by ROUND(jr.lat,4)
  ,      ROUND(jr.lng,4)
  ,      jr.postal_code
  ,      pc.city
  ,      jr.metro_id
  ,      jr.category_id
order by num_recos desc
--)

"""

df = sqlExec(all_sf_sql)
df.opportunity = df.opportunity.astype(float)

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

df_all = df.copy()
df_all = df_all.query("category_id in ('113') ")
hex_col = 'hex_id'
metric_col = 'opportunity'
resolution = 5

resolution = 6
df_c = df.copy()
df_c = df_c.query('metro_id == 1053')
df_c['resolution'] = resolution
    
#find hexes
df_c[hex_col] = df_c.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,x.resolution),1)
    
#aggregate the recos
df_hex_g = df_c.groupby(hex_col).sum()[metric_col].to_frame(metric_col).reset_index()
    
#find center of hex for visualization
df_hex_g['lat'] = df_hex_g[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
df_hex_g['lng'] = df_hex_g[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
   
df_hex_g['resolution'] = resolution
    




#metric_col = 'num_recos'
df_all, df_all_g = add_h3_data(df_all,hex_col,metric_col,resolution)    

resolution = 4
df_4, df_4_g = add_h3_data(df_all,hex_col,metric_col,resolution)    
resolution = 5
df_5, df_5_g = add_h3_data(df_all,hex_col,metric_col,resolution)    
resolution = 6
df_6, df_6_g = add_h3_data(df_all,hex_col,metric_col,resolution)    
resolution = 7
df_7, df_7_g = add_h3_data(df_all,hex_col,metric_col,resolution)    
resolution = 8
df_8, df_8_g = add_h3_data(df_all,hex_col,metric_col,resolution)    

del df_5, df_6, df_7, df_8
df_all_g = pd.concat([df_4_g,df_5_g,df_6_g,df_7_g,df_8_g])

resolution = 4

#plot hexes
plot_scatter(df_all_g,metric_col=metric_col,marker='o',figsize=(17,15))
plt.title('sf_bay_area_opportunity')

#kring_smoothing
k = 2
df_all_s = kring_smoothing(df_all_g,hex_col,metric_col=metric_col,k=k)
print('sum sanity check:', df_all_s[metric_col].sum() / df_all_g[metric_col].sum())
plot_scatter(df_all_s, metric_col=metric_col, marker='o',figsize=(17,15))
plt.title('sf_recos: 2-ring average');


#non-smoothed
#from folium.plugins import MarkerCluster
#mc = MarkerCluster()
map_all = folium.Map(location=[df_all_g.lat.mean(),df_all_g.lng.mean()],zoom_start=5,control_scale=(True),tiles="cartodbpositron")
#map_all = folium.Map(location=[df_hex_g.lat.mean(),df_hex_g.lng.mean()],zoom_start=5,control_scale=(True),tiles="cartodbpositron")
#for index, location_info in df_all_g.iterrows():
#    folium.Marker([location_info["lat"], location_info["lng"]], tooltip=location_info["opportunity"]).add_to(mc)
#mc.add_to(map_all)
map_all = choropleth_map(df_all_g,metric_col,initial_map=map_all)
#map_all = choropleth_map(df_hex_g,metric_col,initial_map=map_all)
map_all.save('us_opportunity_map_'+str(resolution)+'.html')
#map_all.save('us_reco_map.html')
#map_all.save('us_opportuntiy_map_sf_9.html')

#smoothed
map_all = folium.Map(location=[df_all_s.lat.mean(),df_all_s.lng.mean()],zoom_start=6,control_scale=(True),tiles="cartodbpositron")
map_all = choropleth_map(df_all_s,metric_col,initial_map=map_all)
map_all.save('us_smoothed_'+metric_col+'_map_'+str(resolution)+'.html')

#uncompact exploration
resolution = 4
df_hex_g = df_4_g.copy()

quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                         n_quantiles = 100,
                                                         random_state=0)
df_hex_g['reco_transform'] = quantile_transformer.fit_transform(np.array(df_hex_g.num_recos).reshape(-1,1))

df_big = df_all_g[df_all_g['num_recos'] > 10000]

idx1 = (df_hex_g['reco_transform'] >= 1.5 )

df_big = df_hex_g.loc[idx1]
hex_list = list(df_hex_g.loc[~idx1]['hex_id'])
hex_list.extend(h3.uncompact(df_big['hex_id'],resolution + 1))
df_hex_list = pd.DataFrame({'hex_id':hex_list})
df_hex_g = pd.merge(left=df_all_g,right=df_hex_list,how='inner',on='hex_id')

map_all = folium.Map(location=[df_hex_g.lat.mean(),df_hex_g.lng.mean()],zoom_start=5,control_scale=(True),tiles="cartodbpositron")
map_all = choropleth_map(df_hex_g,metric_col,initial_map=map_all)
map_all.save('us_reco_map.html')

df_hex_g['reco_transform'] = quantile_transformer.fit_transform(np.array(df_hex_g.num_recos).reshape(-1,1))
idx2 = (df_hex_g['reco_transform'] >= 1.96 )
df_big = df_hex_g.loc[idx2]

hex_list = list(df_hex_g.loc[~idx2]['hex_id'])
hex_list.extend(h3.uncompact(df_big['hex_id'],resolution + 2))
df_hex_list = pd.DataFrame({'hex_id':hex_list})
df_hex_g = pd.merge(left=df_all_g,right=df_hex_list,how='inner',on='hex_id')

map_all = folium.Map(location=[df_hex_g.lat.mean(),df_hex_g.lng.mean()],zoom_start=5,control_scale=(True),tiles="cartodbpositron")
map_all = choropleth_map(df_hex_g,metric_col,initial_map=map_all)
map_all.save('us_reco_map.html')


df_hex_g['reco_transform'] = quantile_transformer.fit_transform(np.array(df_hex_g.num_recos).reshape(-1,1))
idx3 = (df_hex_g['reco_transform'] >= 2.42 )
df_big = df_hex_g.loc[idx3]

hex_list = list(df_hex_g.loc[~idx3]['hex_id'])
hex_list.extend(h3.uncompact(df_big['hex_id'],resolution + 3))
df_hex_list = pd.DataFrame({'hex_id':hex_list})
df_hex_g = pd.merge(left=df_all_g,right=df_hex_list,how='inner',on='hex_id')

map_all = folium.Map(location=[df_hex_g.lat.mean(),df_hex_g.lng.mean()],zoom_start=5,control_scale=(True),tiles="cartodbpositron")
map_all = choropleth_map(df_hex_g,metric_col,initial_map=map_all)
map_all.save('us_reco_map.html')

#compact exploration
resolution = 8
df_hex_g = df_8_g.copy()
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                         n_quantiles = 100,
                                                         random_state=0)
df_hex_g['reco_transform'] = quantile_transformer.fit_transform(np.array(df_hex_g.num_recos).reshape(-1,1))

df_small = df_all_g[df_all_g['num_recos'] < 1000]

idx1 = (df_hex_g['reco_transform'] < 6)

df_small = df_hex_g.loc[idx1]
hex_list = list(df_hex_g.loc[~idx1]['hex_id'])
hex_list.extend(h3.compact(df_small['hex_id']))
df_hex_list = pd.DataFrame({'hex_id':hex_list})
df_hex_g = pd.merge(left=df_all_g,right=df_hex_list,how='inner',on='hex_id')

map_all = folium.Map(location=[df_hex_g.lat.mean(),df_hex_g.lng.mean()],zoom_start=5,control_scale=(True),tiles="cartodbpositron")
map_all = choropleth_map(df_hex_g,metric_col,initial_map=map_all)
map_all.save('us_reco_map.html')

len(h3.compact(df_small['hex_id']))



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

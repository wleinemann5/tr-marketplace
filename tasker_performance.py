#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:50:07 2021

@author: wilhelmleinemann
"""

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

df = pd.read_csv('tasker_performance_top_metros_jun_aug.csv')

df1 = pd.read_csv('ny_la_tasker_performance_0122.csv')
df2 = pd.read_csv('ny_la_tasker_performance_0222.csv')

df = pd.concat([df1,df2],ignore_index=True)
del df1, df2

df.RECO_MONTH = pd.to_datetime(df['RECO_MONTH'])

#post
df_aug = df[df['RECO_MONTH'] == '2022-02-01']

#pre
df_aug = df[df['RECO_MONTH'] == '2022-01-01']

df_aug['OPPORTUNITY_RANK_ALL'] = df_aug['INVOICE_WEIGHTED_RECOS'].rank(pct=True)

sns.histplot(data=df_aug,
             x='OPPORTUNITY_RANK_ALL',
             y='INVOICE_WEIGHTED_RECOS')

sns.histplot(data=df_aug,
             x='OPPORTUNITY_RANK_ALL',
             y='NUM_INVOICES')

sns.histplot(data=df_aug,
             x='OPPORTUNITY_RANK_ALL',
             y='EXCESS_INVOICES')

df_aug_mc = (df_aug.query('CATEGORY_NAME == "Furniture Assembly"').
             query('METRO_NAME == "LA & OC"'))

title = 'LA Furniture Assembly Tasker Performance Post Exp'
title = 'LA Furniture Assembly Tasker Performance Pre Exp'
#df_aug_mc = df_aug_mc[df_aug_mc['NUM_INVOICES'] > 0]
#df_aug_mc = df_aug_mc[df_aug_mc['NUM_RECOS'] > 100]
df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] = df_aug_mc['INVOICE_WEIGHTED_RECOS'].rank(pct=True)
df_aug_mc['OPPORTUNITY_RANK_ALL'] = df_aug_mc['INVOICE_WEIGHTED_RECOS'].rank()

df_aug_mc['GINI_1'] = (df_aug_mc.OPPORTUNITY_RANK_ALL.max() + 1 - df_aug_mc.OPPORTUNITY_RANK_ALL)*df_aug_mc['INVOICE_WEIGHTED_RECOS']
gini_numerator = df_aug_mc.OPPORTUNITY_RANK_ALL.max() + 1 - 2*(df_aug_mc['GINI_1'].sum()/df_aug_mc['INVOICE_WEIGHTED_RECOS'].sum())
gini_numerator/df_aug_mc.OPPORTUNITY_RANK_ALL.max()

#df_aug_mc['GINI_1'].sum()
#df_aug_mc['INVOICE_WEIGHTED_RECOS'].sum()


df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.95].sum()['EXCESS_INVOICES']
df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] < 0.95].sum()['EXCESS_INVOICES']
df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.95].sum()['EXCESS_REVENUE']
df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] < 0.95].sum()['EXCESS_REVENUE']

#df_aug_mc = df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.975]

df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.95].sum()['NUM_RECOS']/df_aug_mc['NUM_RECOS'].sum()
df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.95].sum()['NUM_INVOICES']/df_aug_mc['NUM_INVOICES'].sum()
df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.95].sum()['INVOICE_WEIGHTED_RECOS']/df_aug_mc['INVOICE_WEIGHTED_RECOS'].sum()
df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.95].sum()['EXCESS_INVOICES']/df_aug_mc['EXCESS_INVOICES'].sum()
df_aug_mc[df_aug_mc['OPPORTUNITY_RANK_ALL_PCT'] >= 0.95].sum()['EXCESS_REVENUE']/df_aug_mc['EXCESS_REVENUE'].sum()

df_aug_mc[df_aug_mc['EXCESS_INVOICES'] >= 10].median()['AVG_RECOMMENDED_PRICE_DELTA']
df_aug_mc[df_aug_mc['EXCESS_INVOICES'] <= 10].median()['AVG_RECOMMENDED_PRICE_DELTA']

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1,1,figsize=(12,8))
sns.histplot(data=df_aug_mc,
             x='OPPORTUNITY_RANK_ALL_PCT',
             y='INVOICE_WEIGHTED_RECOS',
             stat="probability",
             cumulative=False,
             multiple='fill',
             kde=True,
             fill=True,
             element='bars',
             bins=100)
plt.title(title)

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1,1,figsize=(12,8))
sns.histplot(data=df_aug_mc,
             x='OPPORTUNITY_RANK_ALL_PCT',
             y='NUM_INVOICES',
             stat="density",
             cumulative=False,
             kde=True,
             fill=True,
             element='bars',
             bins=100)
plt.title(title)

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1,1,figsize=(12,8))
sns.histplot(data=df_aug_mc,
             x='OPPORTUNITY_RANK_ALL_PCT',
             y='EXCESS_INVOICES',
#             alpha=.2,
             bins=100)
plt.ylim(-30,30)
plt.title(title)

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1,1,figsize=(12,8))
sns.histplot(data=df_aug_mc,
             x='OPPORTUNITY_RANK_ALL_PCT',
             y='EXCESS_REVENUE',
             bins=100)
plt.title(title)



sns.histplot(data=df_aug_mc,
             x='OPPORTUNITY_RANK_ALL_PCT',
             y='NUM_RECOS',
             bins=100)
plt.title(title)

df_tasker = df[df['USER_ID']==5583261]
df_tasker = df[df['USER_ID']==10300994]


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:51:40 2021

@author: wilhelmleinemann
"""

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import theano

import matplotlib.pyplot as plt


%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))

model.basic_RVs

#df = pd.read_csv('met_cat_recos_invoices.csv')
df = pd.read_csv('ywr_mounting_tasker_performance_111521.csv')
df['RECO_MONTH'] = pd.to_datetime(df['RECO_MONTH'])
cols = ['RECO_MONTH','METRO_NAME','CATEGORY_NAME','USER_ID','NUM_RECOS','NUM_INVOICES','INVOICE_WEIGHTED_RECOS']
df = df[cols]
df['metcat'] = df['METRO_NAME'] + '_' +  df['CATEGORY_NAME']

#bound ratio between 0 & 1
def increase_iwr(row):
    if row['NUM_INVOICES'] > row['INVOICE_WEIGHTED_RECOS']:
        return row['NUM_INVOICES']
    else:
        return row['INVOICE_WEIGHTED_RECOS']

def decrease_invoices(row):
    if row['NUM_INVOICES'] > row['INVOICE_WEIGHTED_RECOS']:
        return row['INVOICE_WEIGHTED_RECOS']
    else:
        return row['NUM_INVOICES']

    
df['INVOICE_WEIGHTED_RECOS'] = df.apply(increase_iwr, axis=1)
df['NUM_INVOICES'] = df.apply(decrease_invoices, axis=1)    

#total pooling
#cols2 = ['RECO_MONTH','USER_ID','NUM_INVOICES','INVOICE_WEIGHTED_RECOS']
df_pool = df
df_aug = df_pool[df_pool.RECO_MONTH.dt.month == 8]
#df_sep = df[df.RECO_MONTH.dt.month == 9]
#df_oct = df[df.RECO_MONTH.dt.month == 10]

#look at population mean estimate
df_aug['NUM_INVOICES'].sum()/df_aug['INVOICE_WEIGHTED_RECOS'].sum()

#create simple bayesian model
#cut down the data
df_aug = df_aug.sample(1000,random_state=123123)
N = len(df_aug.NUM_INVOICES)

coords = {
    "obs_id": df_aug.USER_ID,
    "param": ["alpha", "beta"],
}

with pm.Model(coords=coords) as model:
    # Uninformative prior for alpha and beta
    #n_val = pm.Data("n_val", recos)
    recos = pm.Data("recos", value=df_aug.INVOICE_WEIGHTED_RECOS)
    invoices = pm.Data("invoices", value=df_aug.NUM_INVOICES)
    ab = pm.HalfNormal("ab", sigma=10, dims="param")
#    pm.Potential("p(a, b)", logp_ab(ab))

#    X = pm.Deterministic("X", tt.log(ab[0] / ab[1]))
#    Z = pm.Deterministic("Z", tt.log(tt.sum(ab)))

    theta = pm.Beta("theta", alpha=ab[0], beta=ab[1], dims="obs_id")

    p = pm.Binomial("y", p=theta, observed=invoices, n=recos)
    trace = pm.sample(target_accept=0.95, return_inferencedata=True)

#heirarchal partial pooling

N = len(df_aug.NUM_INVOICES)

with pm.Model() as baseball_model:

    phi = pm.Uniform("phi", lower=0.0, upper=1.0)

    kappa_log = pm.Exponential("kappa_log", lam=1.5)
    kappa = pm.Deterministic("kappa", tt.exp(kappa_log))

    thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, shape=N)
    y = pm.Binomial("y", n=df_aug.INVOICE_WEIGHTED_RECOS.values, p=thetas, observed=df_aug.NUM_INVOICES.values)

    
with baseball_model:
    #trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.95, return_inferencedata=True)
    trace = pm.sample(target_accept=0.95, return_inferencedata=True)
    
    # check convergence diagnostics
    assert all(az.rhat(trace) < 1.03)


df_gb_metcat = df.groupby(by=['METRO_NAME','CATEGORY_NAME','RECO_MONTH'],as_index=False).sum()
df_gb_metcat['invoice_ratio'] = df_gb_metcat['NUM_INVOICES']/df_gb_metcat['NUM_RECOS']
df_gb_metcat['wt_invoice_ratio'] = df_gb_metcat['NUM_INVOICES']/df_gb_metcat['INVOICE_WEIGHTED_RECOS']
cols = 
df_gb_month = df.groupby(by='RECO_MONTH',as_index=False).sum()

df_gb_month['invoice_ratio'] = df_gb_month['NUM_INVOICES']/df_gb_month['NUM_RECOS']

trials = [d for d in df_gb_month.NUM_RECOS]
successes = [d for d in df_gb_month.NUM_INVOICES]
with pm.Model() as model:
    trials = [d for d in df_gb_month.NUM_RECOS]
    successes = [d for d in df_gb_month.NUM_INVOICES]
    p = pm.Beta('p', alpha=.3, beta=1.7)
    obs = pm.Binomial('y', n=trials, p=p, shape=30, observed=successes)
    trace = pm.sample(return_inferencedata=True)
    
az.plot_posterior(trace,hdi_prob=0.99)

with model:
    prior_checks = pm.sample_prior_predictive()
    idata_prior = az.from_pymc3(prior=prior_checks)

with model:    
    trace_i = pm.sample()
    idata = az.from_pymc3(trace_i)
    
az.summary(idata_prior,round_to=4)
az.summary(idata,round_to=4)

#rat example
#y is 'successes' per trial
#n is 'number of rats' per trial
#N is number of trials

y = np.array([
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
    1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  5,  2,
    5,  3,  2,  7,  7,  3,  3,  2,  9, 10,  4,  4,  4,  4,  4,  4,  4,
    10,  4,  4,  4,  5, 11, 12,  5,  5,  6,  5,  6,  6,  6,  6, 16, 15,
    15,  9,  4
])

invoices = df_gb_month.NUM_INVOICES.values
recos = df_gb_month.NUM_RECOS.values
N = len(recos)

n = np.array([
    20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20,
    20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19,
    46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20,
    48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46,
    47, 24, 14
])
# fmt: on

N = len(n)

coords = {
    "obs_id": np.arange(N),
    "param": ["alpha", "beta"],
}

def logp_ab(value):
    """ prior density"""
    return tt.log(tt.pow(tt.sum(value), -5 / 2))


with pm.Model(coords=coords) as model:
    # Uninformative prior for alpha and beta
    #n_val = pm.Data("n_val", recos)
    recos = pm.Data("recos", value=df_gb_month.NUM_RECOS)
    invoices = pm.Data("invoices", value=df_gb_month.NUM_INVOICES)
    ab = pm.HalfNormal("ab", sigma=10, dims="param")
#    pm.Potential("p(a, b)", logp_ab(ab))

#    X = pm.Deterministic("X", tt.log(ab[0] / ab[1]))
#    Z = pm.Deterministic("Z", tt.log(tt.sum(ab)))

    theta = pm.Beta("theta", alpha=ab[0], beta=ab[1], dims="obs_id")

    p = pm.Binomial("y", p=theta, observed=invoices, n=recos)
    trace = pm.sample(target_accept=0.95, return_inferencedata=True)
    

# Check the trace. Looks good!
az.plot_trace(trace, var_names=["ab"], compact=False)

az.plot_trace(trace, var_names=["phi"], compact=False)

az.plot_posterior(trace, var_names=["ab"]);
az.plot_posterior(trace, var_names=["theta"]);
az.plot_posterior(trace, var_names=["thetas"]);

az.plot_forest(trace, var_names=["theta"]);
az.plot_forest(trace, var_names=["thetas"], r_hat=True);

tp_post_mean = trace.posterior.mean(('chain','draw'))
tp_post_mean.theta

pp_post_mean = trace.posterior.mean(('chain','draw'))
pp_post_mean.thetas


samples = pm.sample_posterior_predictive(trace, samples=1000, model=model)



# estimate the means from the samples
trace.posterior["ab"].mean(("chain", "draw")).values


#herirarchal logisitic regression
df_aug['CATEGORY_NAME'] = df_aug.CATEGORY_NAME.astype('category')
category_idx = df_aug.CATEGORY_NAME.cat.codes.values
category_names = df_aug.CATEGORY_NAME.unique()

#n_taskers = df_aug.USER_ID.nunique()
n_taskers = df_aug.USER_ID.count()
n_categories = category_names.size

with pm.Model() as hierarchical_model:
    iwr = pm.Data('iwr',df_aug.INVOICE_WEIGHTED_RECOS.values)
    invoices = pm.Data('invoices',df_aug.NUM_INVOICES.values)
    
    #hyper parameters
    aSigma = pm.Gamma('aSigma',1.64,0.32)
    a0 = pm.Normal('a0',0.0, tau=1/2**2)
    a =  pm.Normal('a',0.0,tau=1/aSigma**2, shape=n_categories)
    
    #parameteres for categories 
    omega = pm.Deterministic('omega', pm.invlogit(a0 + a))
    kappa = pm.Gamma('kappa',0.01,0.01)
    
    #parameters for individual taskers
    mu = pm.Beta('mu',
                 omega[category_idx]*kappa+1,
                 (1-omega[category_idx])*kappa+1,
                 shape = n_taskers)
    
    y = pm.Binomial('y', n=iwr, p = mu, observed = invoices)
    
    #convert a0, a to sum-to-zero b0, b
    m = pm.Deterministic('m', a0 + a)
    b0 = pm.Deterministic('b0', tt.mean(m))
    b = pm.Deterministic('b', m - b0)
    
with hierarchical_model:
    trace3 = pm.sample(1000, cores=4,target_accept=0.95, return_inferencedata=True)
    
az.plot_trace(trace3)
az.plot_posterior(trace3)
az.plot_forest(trace3)

data_mu = pm.summary(trace3,'mu')

data_mu['sharpe'] = (data_mu['mean']- .18)/data_mu['sd']

out_of_sample_iwr = np.full((1000),100)
with hierarchical_model:
    pm.set_data({'iwr':out_of_sample_iwr})
    ppc = pm.sample_posterior_predictive(trace3)
    model_preds = ppc['y']
    
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=hierarchical_model));

az.summary(trace3,var_names=['mu'],round_to=2)
    
preds = pd.DataFrame(model_preds)
preds.mean().sort_values()


#herirarchal logisitic regression w/multiple categories
df_aug['CATEGORY_NAME'] = df_aug.CATEGORY_NAME.astype('category')
df_aug['METRO_NAME'] = df_aug.METRO_NAME.astype('category')

df_aug['CATEGORY_NAME']
category_idx = df_aug.CATEGORY_NAME.cat.codes.values
category_names = df_aug.CATEGORY_NAME.unique()
metro_idx = df_aug.METRO_NAME.cat.codes.values
metro_names = df_aug.METRO_NAME.unique()

df_aug['f_category_name'], Ncategory = df_aug['CATEGORY_NAME'].factorize()
df_aug['f_metro_name'], Nmetros = df_aug['METRO_NAME'].factorize()

x_n = ['f_category_name','f_metro_name']
x_1 = df_aug[x_n].values
cat = df_aug['f_category_name'].values
met = df_aug['f_metro_name'].values

#n_taskers = df_aug.USER_ID.nunique()
n_taskers = df_aug.USER_ID.count()
n_categories = category_names.size
n_metros = metro_names.size

with pm.Model() as pooled_categorical_model:
  x1 = theano.shared(data['x1'].values)
  x2 = theano.shared(data['x2'].values)
  y1 = pm.Data('y1',data['y1'].values)

  #hyper priors for cat2
  pooled_x2_mu = pm.Normal('pooled_x2_mu',0,1)
  pooled_x2_sigma = pm.Exponential('pooled_x2_sigma',lam=1)

  #hyper priors for cat1
  pooled_x1_mu = pm.Normal('pooled_x1_mu',0,1)
  pooled_x1_sigma = pm.Exponential('pooled_x1_sigma',lam=1)

  cat_x1 = pm.Normal('cat_x1',pooled_x1_mu,pooled_x2_sigma,shape=len(data['x1'].unique()))
  cat_x2 = pm.Normal('cat_x2',pooled_x2_mu,pooled_x2_sigma,shape=len(data['x2'].unique()))
  sigma = pm.Exponential('error',lam=1)

  mu = cat_x1[x1] + cat_x2[x2]
  y_hat = pm.Normal('y_hat',mu,sigma,observed=y1)
  trace_pooled_categorical = pm.sample(500,tune=1500,chains=2, cores=2, target_accept=0.95)


#this model needs tuning
#may need to center
with pm.Model() as hierarchical_model_2:
    iwr = pm.Data('iwr',df_aug.INVOICE_WEIGHTED_RECOS.values)
    invoices = pm.Data('invoices',df_aug.NUM_INVOICES.values)
    
    #hyper-priors
    s_c = pm.HalfNormal('s_c',sd=5)
    s_m = pm.HalfNormal('s_m',sd=5)
    
    #priors
    b0 = pm.Normal('b0',0,1.5)
    b_cat = pm.Normal('b_cat',0.0,s_c,shape=n_categories)
    b_met = pm.Normal('b_met',0.0,s_m,shape=n_metros)
    
    #parameteres for categories 
    #mu = pm.math.invlogit(b0 + b_cat[cat] + b_met[met])
    omega = pm.Deterministic('omega',pm.math.invlogit(b0 + b_cat[cat] + b_met[met]))
    kappa = pm.Gamma('kappa',0.01,0.01)
    
    #parameters for individual taskers
    mu = pm.Beta('mu',
                 omega*kappa+1,
                 (1-omega)*kappa+1,
                 shape = n_taskers)
  
    y = pm.Binomial('y', n=iwr, p = mu, observed = invoices)
    

with hierarchical_model_2:
    trace3 = pm.sample(1000, cores=4,target_accept=0.95, return_inferencedata=True)
    
az.summary(trace3)
az.plot_forest(trace3,var_names=['mu'],combined=True,
               kind='ridgeplot',
               coords={'mu_dim_0':range(540,560)},
               ridgeplot_truncate=False,
#               colors='white',
               ridgeplot_overlap = 3,
               ridgeplot_alpha=0.5
               )
sum_df = az.summary(trace3,var_names=['mu'])
sum_df = az.summary(trace3,var_names=['b_cat','b_met'])
    
out_of_sample_iwr = np.full((1000),100)
with hierarchical_model_2:
    pm.set_data({'iwr':out_of_sample_iwr})
    ppc = pm.sample_posterior_predictive(trace3)
    model_preds = ppc['y']

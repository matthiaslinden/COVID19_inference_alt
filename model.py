#!/usr/bin/env python3.7
#coding:utf-8

# Heavily borrows from https://github.com/Priesemann-Group/covid19_inference

import numpy as np
import numpy.ma as ma
import theano
import theano.tensor as tt
import theano.tensor.signal.conv as tt_conv
import pymc3 as pm

import pickle
import datetime
from ModelInfo import Modelinfo,Datasets

import matplotlib.pyplot as plt

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
#theano.config.floatX = "float64"
#theano.config.compute_test_value = "ignore"
#theano.config.exception_verbosity="high"
theano.config.optimizer='fast_run'  
#theano.config.mode='DebugMode'  

# def lambda_t_with_sigmoids(
#     change_points_list, pr_median_lambda_0, pr_sigma_lambda_0=0.5, model=None
# ):
#     """
#
#     Parameters
#     ----------
#     change_points_list
#     pr_median_lambda_0
#     pr_sigma_lambda_0
#     model : :class:`Cov19Model`
#         if none, it is retrieved from the context
#
#     Returns
#     -------
#
#     """
#
#     model = modelcontext(model)
#     model.sim_shape = model.sim_shape
#
#     lambda_list, tr_time_list, tr_len_list = make_change_point_RVs(
#         change_points_list, pr_median_lambda_0, pr_sigma_lambda_0, model=model
#     )
#
#     # model.sim_shape = (time, state)
#     # build the time-dependent spreading rate
#     if len(model.sim_shape) == 2:
#         lambda_t_list = [lambda_list[0] * tt.ones(model.sim_shape)]
#     else:
#         lambda_t_list = [lambda_list[0] * tt.ones(model.sim_shape)]
#     lambda_before = lambda_list[0]
#
#     for tr_time, tr_len, lambda_after in zip(
#         tr_time_list, tr_len_list, lambda_list[1:]
#     ):
#         t = np.arange(model.sim_shape[0])
#         tr_len = tr_len + 1e-5
#         if len(model.sim_shape) == 2:
#             t = np.repeat(t[:, None], model.sim_shape[1], axis=-1)
#         lambda_t = tt.nnet.sigmoid((t - tr_time) / tr_len * 4) * (
#             lambda_after - lambda_before
#         )
#         # tr_len*4 because the derivative of the sigmoid at zero is 1/4, we want to set it to 1/tr_len
#         lambda_before = lambda_after
#         lambda_t_list.append(lambda_t)
#     lambda_t_log = sum(lambda_t_list)
#
#     pm.Deterministic("lambda_t", tt.exp(lambda_t_log))
#
#     return lambda_t_log
#
# def make_change_point_RVs(
#     change_points_list, pr_median_lambda_0, pr_sigma_lambda_0=1, model=None
# ):
#     """
#
#     Parameters
#     ----------
#     priors_dict
#     change_points_list
#     model
#
#     Returns
#     -------
#
#     """
#
#     default_priors_change_points = dict(
#         pr_median_lambda=pr_median_lambda_0,
#         pr_sigma_lambda=pr_sigma_lambda_0,
#         pr_sigma_date_transient=2,
#         pr_median_transient_len=4,
#         pr_sigma_transient_len=0.5,
#         pr_mean_date_transient=None,
#     )
#
#     for cp_priors in change_points_list:
#         mh.set_missing_with_default(cp_priors, default_priors_change_points)
#
#     model = modelcontext(model)
#     len_L2 = () if model.sim_ndim == 1 else model.sim_shape[1]
#
#     lambda_log_list = []
#     tr_time_list = []
#     tr_len_list = []
#
#     #
#     lambda_0_L2_log, lambda_0_L1_log = hierarchical_normal(
#         "lambda_0_log",
#         "sigma_lambda_0",
#         np.log(pr_median_lambda_0),
#         pr_sigma_lambda_0,
#         len_L2,
#         w=0.4,
#         error_cauchy=False,
#     )
#     if lambda_0_L1_log is not None:
#         pm.Deterministic("lambda_0_L2", tt.exp(lambda_0_L2_log))
#         pm.Deterministic("lambda_0_L1", tt.exp(lambda_0_L1_log))
#     else:
#         pm.Deterministic("lambda_0", tt.exp(lambda_0_L2_log))
#
#     lambda_log_list.append(lambda_0_L2_log)
#     for i, cp in enumerate(change_points_list):
#         if cp["pr_median_lambda"] == "previous":
#             pr_sigma_lambda = lambda_log_list[-1]
#         else:
#             pr_sigma_lambda = np.log(cp["pr_median_lambda"])
#         lambda_cp_L2_log, lambda_cp_L1_log = hierarchical_normal(
#             f"lambda_{i + 1}_log",
#             f"sigma_lambda_{i + 1}",
#             pr_sigma_lambda,
#             cp["pr_sigma_lambda"],
#             len_L2,
#             w=0.7,
#             error_cauchy=False,
#         )
#         if lambda_cp_L1_log is not None:
#             pm.Deterministic(f"lambda_{i + 1}_L2", tt.exp(lambda_cp_L2_log))
#             pm.Deterministic(f"lambda_{i + 1}_L1", tt.exp(lambda_cp_L1_log))
#         else:
#             pm.Deterministic(f"lambda_{i + 1}", tt.exp(lambda_cp_L2_log))
#
#         lambda_log_list.append(lambda_cp_L2_log)
#
#     dt_before = model.sim_begin
#     for i, cp in enumerate(change_points_list):
#         dt_begin_transient = cp["pr_mean_date_transient"]
#         if dt_before is not None and dt_before > dt_begin_transient:
#             raise RuntimeError("Dates of change points are not temporally ordered")
#         prior_mean = (dt_begin_transient - model.sim_begin).days
#         tr_time_L2, _ = hierarchical_normal(
#             f"transient_day_{i + 1}",
#             f"sigma_transient_day_{i + 1}",
#             prior_mean,
#             cp["pr_sigma_date_transient"],
#             len_L2,
#             w=0.5,
#             error_cauchy=False,
#             error_fact=1.0,
#         )
#
#         tr_time_list.append(tr_time_L2)
#         dt_before = dt_begin_transient
#
#     for i, cp in enumerate(change_points_list):
#         # if model.sim_ndim == 1:
#         tr_len_L2_log, tr_len_L1_log = hierarchical_normal(
#             f"transient_len_{i + 1}_log",
#             f"sigma_transient_len_{i + 1}",
#             np.log(cp["pr_median_transient_len"]),
#             cp["pr_sigma_transient_len"],
#             len_L2,
#             w=0.7,
#             error_cauchy=False,
#         )
#         if tr_len_L1_log is not None:
#             pm.Deterministic(f"transient_len_{i + 1}_L2", tt.exp(tr_len_L2_log))
#             pm.Deterministic(f"transient_len_{i + 1}_L1", tt.exp(tr_len_L1_log))
#         else:
#             pm.Deterministic(f"transient_len_{i + 1}", tt.exp(tr_len_L2_log))
#
#         tr_len_list.append(tt.exp(tr_len_L2_log))
#     return lambda_log_list, tr_time_list, tr_len_list

def tt_lognormal(x, mu, sigma):
# Limit to prevent NANs
    x = tt.clip(x,1e-9,1e12)
    sigma = tt.clip(sigma,1e-9,1e12)
    mu = tt.clip(mu,1e-9,1e12)
    
    distr = 1/x * tt.exp( -( (tt.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return distr / (tt.sum(distr, axis=0) + 1e-12)

def GenInit(l,a1,a2,t1=10,t2=27,offset=8):
    x = tt.arange(l)
    d1 = tt_lognormal(x,tt.log(t1),.8)*2350 #.4 / 23500
    d2 = tt_lognormal(x,tt.log(t2),.25)*12500
    
    
    din = d1*a1 + d2*a2
    din = tt.roll(din,-offset)
    din = tt.set_subtensor(din[-offset:],0.)
 #   din = tt.set_subtensor(din[0],10)
    return din
    
    
    
# def reportDelayDist(x, mu1, sig1, mu2, sig2, r ):
#     """ builds the reporting Delay distribution from two lognormal distributions"""
# #    x = tt.clip(x,1e-12,1e12) # Checks to improve stability and resilience against NANs
#     r = tt.clip(r,0,1)
#     xm = tt.alloc(x,r.shape[0],x.shape[0])
#     d1 = tt_lognormal(x,tt.log(mu1),sig1) * xm
#     d2 = tt_lognormal(x,tt.log(mu2),sig2) * (1-xm)
#     ds = d1+d2
#     return ds / (tt.sum(ds,axis=0) + 1e-12)
#
# def DelayedReporting(cases,mu1,sig1,mu2,sig2,r,n=64):
#     cases = tt.cast(cases,'float64')
#     x = tt.arange(1,n+1)
#     dist = reportDelayDist(x,mu1,sig1,mu2,sig2,r)
#
#     mc = tt.nlinalg.alloc_diag(cases)
#   #  dist0 = tt.alloc(0.,1,dist.shape[0])
#   #  dist0d = tt.cast(dist0,'float64')
#     distm = tt.set_subtensor(dist0d[0,:],dist)
#
#     mcc = tt.signal.conv.conv2d(mc,distm,border_mode='full')
#     ds = tt.sum(mcc,axis=1)
#
#     return ds,mcc,dist

def reportDelayDistFunc(cases,mu1,sig1,mu2,sig2,r,n):
    """ from onset of illnesses 'cases', generate a time series for every day of infection representing when the cases are reported. """
    m1 = tt.cast(mu1,'float64')
    s1 = tt.cast(sig1,'float64')
    m2 = tt.cast(mu2,'float64')
    s2 = tt.cast(sig2,'float64')
    sr = tt.cast(r,'float64')
    n = tt.cast(n,'int64')
    x = tt.arange(1,n+1)
    
    # Prepare the Distributions
#    d = tt.clip(tt.cast(d,'float64'),1e-12,1e12) # Checks to improve stability and resilience against NANs
    sr = tt.clip(r,1e-12,1-1e-12)
    d1 = tt_lognormal(x,tt.log(m1),s1)
    d2 = tt_lognormal(x,tt.log(m2),s2)
    
    d1 = tt.alloc(d1,1,d1.shape[0])
    d2 = tt.alloc(d2,1,d2.shape[0])
    # Prepare cases as diagonal of matrix
    cin = tt.cast(cases,'float64')
    c2d = tt.nlinalg.alloc_diag(cin)
    # Create a Vector
    
    cf1 = tt.signal.conv.conv2d(c2d,d1,border_mode='full')
    cf2 = tt.signal.conv.conv2d(c2d,d2,border_mode='full')
    
    cfo = (sr*cf1.T + (tt.ones_like(sr)-sr)*cf2.T).T
    return cfo#

def conv_offset(inp,filt,amplitude=1.):
#    offset = tt.cast(offset,'int64')
    amplitude = tt.cast(amplitude,'float64')
    amplitude = tt.clip(amplitude,1e-12,1e9) # Limit to prevent NANs
    
    zero = tt.zeros_like(inp)
    a0rp = tt.concatenate((inp,zero,),0) * amplitude
#    a0rp = tt.set_subtensor(a0,0.) * amplitude
    
    a0rp3d = tt.alloc(0.,1,a0rp.shape[0],1 )
    a0rp = tt.set_subtensor(a0rp3d[0,:,0],a0rp)
    filt3d = tt.alloc(0.,1,filt.shape[0],1 )
    filt = tt.set_subtensor(filt3d[0,:,0],filt)
    return tt_conv.conv2d(a0rp,filt,None,None,border_mode='full').flatten()
    
def DelayLognormal(infected_t,median,sigma,factor,l=40):
    l = tt.cast(l,'int64')
    beta = tt_lognormal(tt.arange(l*2),tt.log(median),sigma)
    return conv_offset(infected_t,beta,factor)
    
# def conv_offset(inp,filt,amplitude=1,offset=0):
#     offset = tt.cast(offset,'int64')
#     amplitude = tt.clip(amplitude,1e-12,1e9) # Limit to prevent NANs
#
#     zero = tt.zeros_like(inp)
#     a0 = tt.concatenate((inp,zero,),0)
#     a0r = tt.roll(a0,offset,0)
#     a0rp = tt.set_subtensor(a0r[:offset],0.) * amplitude
#
#     a0rp3d = tt.alloc(0.,1,a0rp.shape[0],1 )
#     a0rp = tt.set_subtensor(a0rp3d[0,:,0],a0rp)
#     filt3d = tt.alloc(0.,1,filt.shape[0],1 )
#     filt = tt.set_subtensor(filt3d[0,:,0],filt)
#     return tt_conv.conv2d(a0rp,filt,None,None,border_mode='full').flatten()
#
# def DelayLognormal(infected_t,median,sigma,factor,l=40,offset=0):
#     beta = tt_lognormal(tt.arange(l*2),tt.log(median),sigma)
#     return conv_offset(infected_t,beta,factor,offset)

def WeeklyRandomWalk(name,n,initial,flt=np.array([1.,1.,1.,1.,1.,1.,1.],dtype=np.float64),sigma=.05,offset=0,shorten=6):
    awl = np.array([0,1,1,1,1,1,1])
    additional_week = awl[offset%7]
  #  offset = tt.cast(offset,'int64')
    delay_list_length = n//7+additional_week-shorten
    rw_list = []
    rw_list.append(initial)
    sigma_random_walk = pm.HalfNormal(name=name+"_sigma_random_walk", sigma=sigma)
    delay_ratio_random_walk = pm.distributions.timeseries.GaussianRandomWalk(
                              name=name+"_random_walk",mu=0,
                              sigma=sigma_random_walk,shape=delay_list_length,
                              init=pm.Normal.dist(sigma=sigma),
                        )
    flt = tt.cast(flt,'float64')
    flt = flt / tt.sum(flt)
    val = delay_ratio_random_walk
     
    # Derive longer weekly list to allow filtering, offset(rotation) and same-valued tail (shorten > 0)
    lval = tt.alloc(0.,val.shape[0]+4+shorten)  # 2 weeks at the front, 2 weeks at the back
    lval = tt.cast(lval,"float64")
    lval = tt.set_subtensor(lval[2:-(2+shorten)],val) # 
    lval = tt.set_subtensor(lval[:2],val[0])
    lval = tt.set_subtensor(lval[-(2+shorten):],val[-1]) # extend the end

    # Create daily vector by flattening a weeks x 7days matrix
    m = tt.alloc(lval,7,lval.shape[0])
    mf = tt.flatten(m.T,ndim=1)
    mf = tt.roll(mf,-offset)
    
    # Prepare as stack of 'images' and 'filters' for conv2D
    mf2 = tt.alloc(mf,1,mf.shape[0])
    kern2 = tt.alloc(flt,1,flt.shape[0])
    
    r = tt.signal.conv.conv2d(mf2,kern2,border_mode='full')
    r = tt.roll(r[0],offset)
    rs = r[(14+flt.shape[0]//2):(n+14+flt.shape[0]//2)]
   
    return rw_list[0]+rs

def WeeklyRandomWalkWeekend(name,n,initial,wfactor,flt=np.array([.05,.1,.7,.1,.05],dtype=np.float64),sigma=.05,offset=0):
    awl = np.array([0,1,1,1,1,1,1])
    additional_week = awl[offset%7]   # if firstday == monday, no additional week is needed
    additional_week = 1
    walk_len = n//7+additional_week
    rw_list = []
    rw_list.append(initial)
    # Generate "stepsize"
    sigma_random_walk = pm.HalfNormal(name=name+"_sigma_random_walk", sigma=sigma)
    random_walk = pm.distributions.timeseries.GaussianRandomWalk(
                              name=name+"_random_walk",mu=0,
                              sigma=sigma_random_walk,shape=walk_len,
                              init=pm.Normal.dist(sigma=sigma),
                        )
    flt = flt / tt.sum(flt)
    val = random_walk
  # generates a longer list, 2 at front, two at the back with the same vaule as the original front / back
  # --> 2 weeks pre / post to allow simple filtering and offset of up to one week length eacht.
    lval = tt.alloc(0.,val.shape[0]+4)  # streched list of values
    lval = tt.cast(lval,"float64")
    lval = tt.set_subtensor(lval[2:-2],val)
    lval = tt.set_subtensor(lval[:2],val[0])
    lval = tt.set_subtensor(lval[-2:],val[-1]) # extend the 

    # Generate Matrix 7x(#weeks) shape, which was weekly values dublicated over 7 entries
    m = tt.alloc(lval,7,lval.shape[0])
    mf = tt.flatten(m.T,ndim=1) # Flatten it, now 7 weekly values are 
    
    # Format Matrix 
    mf2 = tt.alloc(mf,1,mf.shape[0])
    kern2 = tt.alloc(flt,1,flt.shape[0])
    
    daily_values = tt.signal.conv.conv2d(mf2,kern2,border_mode='full')
    daily_values = tt.roll(daily_values[0],-offset)
    daily_values_ranged = daily_values[(14+flt.shape[0]//2):(7*val.shape[0]+14+flt.shape[0]//2)][:n]
   
    # Generate 7x(n days) maxtrix marking day of week
    d_oeye = tt.roll(tt.eye(7),-offset,axis=1)
    week_mask = tt.tile(d_oeye,walk_len)[:,:n]
    
    daily_walk = rw_list[0]+daily_values_ranged
    
    # Create Mask with wfactor at the weekends otherwiese 1, then multiply with daily_walk
    weekend_m = week_mask[5] + week_mask[6]   # Saturday + Sunday
    weekend_f = weekend_m*wfactor - weekend_m + tt.ones_like(weekend_m)
    daily_walk = daily_walk * weekend_f
   
    return daily_walk,week_mask

def DailyRandomWalkWeekend(name,n,initial,wfactor,flt=np.array([.05,.1,.7,.1,.05],dtype=np.float64),sigma=.01,offset=0):
    
    rw_list = []
    rw_list.append(initial)
    # Generate "stepsize"
    sigma_random_walk = pm.HalfNormal(name=name+"_sigma_random_walk", sigma=sigma)
    random_walk = pm.distributions.timeseries.GaussianRandomWalk(
                              name=name+"_random_walk",mu=0,
                              sigma=sigma_random_walk,shape=n,
                              init=pm.Normal.dist(sigma=sigma),
                        )

    # Generate Matrix 7x(#weeks) shape, which was weekly values dublicated over 7 entries
    awl = np.array([0,1,1,1,1,1,1])
    additional_week = awl[offset%7]   # if firstday == monday, no additional week is needed
    weeks = n//7+additional_week    
   
    # Generate 7x(n days) maxtrix marking day of week
    d_oeye = tt.roll(tt.eye(7),-offset,axis=1)
    week_mask = tt.tile(d_oeye,weeks)[:,:n]
    
    daily_walk = rw_list[0]+random_walk
    
    # Create Mask with wfactor at the weekends otherwiese 1, then multiply with daily_walk
    weekend_m = week_mask[5] + week_mask[6]   # Saturday + Sunday
    weekend_f = weekend_m*wfactor - weekend_m + tt.ones_like(weekend_m)
    daily_walk = daily_walk * weekend_f
   
    return daily_walk,week_mask

def GenChangepoints():
    change_points = [
        # mild distancing
        dict(
            pr_mean_date_transient=datetime.datetime(2020, 3, 9)
            # account for new implementation where transients_day is centered, not begin
            + datetime.timedelta(days=1.5),
            pr_median_transient_len=3,
            pr_sigma_transient_len=0.3,
            pr_sigma_date_transient=3,
            pr_median_lambda=0.2,
            pr_sigma_lambda=0.5,
        ),
        # strong distancing
        dict(
            pr_mean_date_transient=datetime.datetime(2020, 3, 16)
            + datetime.timedelta(days=1.5),
            pr_median_transient_len=3,
            pr_sigma_transient_len=0.3,
            pr_sigma_date_transient=1,
            pr_median_lambda=1 / 8,
            pr_sigma_lambda=0.5,
        ),
        # contact ban
        dict(
            pr_mean_date_transient=datetime.datetime(2020, 3, 23)
            + datetime.timedelta(days=1.5),
            pr_median_transient_len=3,
            pr_sigma_transient_len=0.3,
            pr_sigma_date_transient=1,
            pr_median_lambda=1 / 16,
            pr_sigma_lambda=0.5,
        ),
        # opening
        dict(
            pr_mean_date_transient=datetime.datetime(2020, 4, 20)
            + datetime.timedelta(days=1.5),
            pr_median_transient_len=3,
            pr_sigma_transient_len=0.3,
            pr_sigma_date_transient=1,
            pr_median_lambda=1 / 8,
            pr_sigma_lambda=0.5,
        ),
        # opening 2
        dict(
            pr_mean_date_transient=datetime.datetime(2020, 5, 4)
            + datetime.timedelta(days=1.5),
            pr_median_transient_len=3,
            pr_sigma_transient_len=0.3,
            pr_sigma_date_transient=1,
            pr_median_lambda=1 / 8,
            pr_sigma_lambda=0.5,
        ),
    ]
    return lambda_t_with_sigmoids(change_points,)

def TransferWeekendReported(r_t,f,mask):
    """ Moves f* value at r_t to r_t+2 / r_t+1 on saturdays and sundays """
    sat = r_t * mask[5] * f # Trnasfer cases
    sut = r_t * mask[6] * f
    r_t = r_t - sat - sut   # Substract the transfered cases
    satr = tt.roll(sat,2)   # Shift the transfered cases
    satr = tt.set_subtensor(satr[:2],0)
    sutr = tt.roll(sut,1)
    sutr = tt.set_subtensor(sutr[:1],0)
    r_t = r_t + satr + sutr # Add up
    return r_t
    
def DelayDaily(data,max_delay,week_mask):
    """ Delay Cases from one day to the next by weekdayfactor """
#    factor = pm.TruncatedNormal(name="d_factor",mu=.1,sigma=.1,lower=-1,upper=1,shape=(max_delay,7,))
    factor = pm.TruncatedNormal(name="d_factor",mu=.01,sigma=.05,lower=0,upper=1,shape=(max_delay,7,))
    
    
    t = tt.dot(factor[0],week_mask)
    tv = data*t
    d_sub = data-tv
    tvr = tt.roll(tv,1)
    tvr = tt.set_subtensor(tvr[:1],0)
    d_add = d_sub + tvr
    
    if max_delay > 1:
        data = d_add
        t = tt.dot(factor[1],week_mask)
        tv = data*t
        d_sub = data-tv
        tvr = tt.roll(tv,2)
        tvr = tt.set_subtensor(tvr[:2],0)
        d_add = d_sub + tvr
    if max_delay > 2:
        data = d_add
        t = tt.dot(factor[2],week_mask)
        tv = data*t
        d_sub = data-tv
        tvr = tt.roll(tv,3)
        tvr = tt.set_subtensor(tvr[:3],0)
        d_add = d_sub + tvr
    
    return d_add

def SEIR_model(N, imported_t,lambda_t, median_incubation,sigma_incubation,l=32):
    N = tt.cast(N,'float64')
    beta = tt_lognormal(tt.arange(l), tt.log(median_incubation), sigma_incubation)
    
    # Dirty hack to prevent nan
 #   beta = tt.alloc(0,l)
  #  beta = tt.set_subtensor(beta[tt.clip(tt.cast(median_incubation,'int32'),1,l-2)],1)
     
    lambda_t = tt.as_tensor_variable(lambda_t)
    imported_t = tt.as_tensor_variable(imported_t)

    def new_day(lambda_at_t,imported_at_t,infected,E_t,beta,N):
        f = E_t / N
     #   f = 1
        new = imported_at_t + theano.dot(infected,beta) * lambda_at_t * f
        new = tt.clip(new,0,N)
     
        infected = tt.roll(infected,1,0)
        infected = tt.set_subtensor(infected[:1],new,inplace=False)
        E_t = tt.clip(E_t-new,0,E_t)
#        E_t = E_t-new
        return new,infected,E_t
    
    outputs_info = [None,np.zeros(l),N]
    infected_t,updates = theano.scan(fn=new_day,
                                     sequences=[lambda_t,imported_t],
                                     outputs_info=outputs_info,
                                     non_sequences=[beta,N],
                                     profile=False)
                                     
    return infected_t


pr_d = {}
#priors_dict[""] = 
pr_d["pr_beta_sigma_obs"] = 10
pr_d["pr_mean_median_incubation"] = 4.
pr_d["sigma_incubation"] = 0.418
pr_d["pr_sigma_random_walk"] = 0.005 #0.05
pr_d["pr_median_lambda_0"] = 3.0
pr_d["pr_sigma_lambda_0"] = .5

# Data
front = [-1]*4
epi_curve = front+[7, 9, 9, 10, 34, 19, 27, 45, 65, 101, 148, 159, 202, 184, 204, 278, 267, 363, 376, 599, 756, 1029, 1557, 1999, 2495, 2754, 3356, 3382, 3528, 4680, 3907, 4038, 3469, 4072, 3288, 2739, 3928, 2912, 3179, 2804, 2897, 2721, 2190, 3117, 2281, 2700, 2414, 2463, 1956, 1736, 2239, 1887, 1761, 1685, 1406, 1190, 1222, 1157, 1187, 1138, 1000, 960, 847, 727, 928, 719, 690, 698, 621, 524, 463, 628, 462, 419, 467, 363, 340, 352, 430, 348, 302, 271, 209, 165, 116, 126, 63, 16, 3, 0,0,0,0]

epi_curve = np.asarray(epi_curve,dtype=np.float64)
m_epi_curve = ma.masked_less_equal(epi_curve,0.,copy=True)


initial = [7,9,8,10,5,0,4,13]+[0]*(len(epi_curve)-8)
imported_cases = np.asarray(initial,dtype=np.float64)

startdate = datetime.date(2020,2,16)-datetime.timedelta(len(front))
mi = Modelinfo(startdate,epi_curve.shape[0])
ds = Datasets()
ds.AddDataSeries("epi_curve",m_epi_curve,startdate)
ds.AddDataSeries("initial",imported_cases,startdate)
print(mi)


hospital = [-1]*17+[-1]*20+[379,352,529,581,661,432,566,719,759,834,839,748,747,500,676,751,791,789,604,468,425,415,644,705,645,660,428,310,405,569,500,490,410,286,160,296,360,379,470,192,132,130]+[-1]*5
hospital_begin = 37
hospital = ma.masked_less(hospital,0.,copy=True)[hospital_begin:]

icu_deaths = [-1]*17+[-1]*36+[93,154,0,131,88,66,228,255,305,123,50,59,108,91,115,39,22,29,24,68,69,52,40,46,85]
icu_deaths_begin = 17+30
icu_deaths = ma.masked_less(icu_deaths,0.,copy=True)[icu_deaths_begin:]

deaths = [-1]*(17+4)+[-1,-1,-1,-1,-1,-1,0,1,2,0,3,4,0,0,0,8,11,16,8,31,28,35,49,55,72,64,66,128,149,140,145,141,184,92,173,254,246,266,171,129,126,170,285,315,299,242,184,110,194,281,215,227,179,140,110,163,202,173,193,94,74,43,139,165,123,147]+[-1]*8
deaths = np.asarray(deaths,dtype=np.float64)
deaths = ma.masked_less(deaths,0.,copy=True)

reported = [-1]*4+[66,138,239,156,107,237,157,271,802,693,733,400,1817,1144,1036,2807,2958,2705,1948,4062,4764,4118,4954,5780,6294,3965,4751,4615,5453,6156,6174,6082,5936,3677,3834,4003,4974,5323,4133,2821,2537,2082,2486,2866,3380,3609,2458,1775,1785,2237,2352,2337,2055,1737,1018,1144,1304,1478,1639,945,793,679,685,947]
reported = ma.masked_less(reported,0.,copy=True)

print("Len of inputs",len(deaths),len(epi_curve),len(imported_cases),len(hospital),print(icu_deaths),len(reported))

#num_days_sim = min(len(epi_curve),len(imported_cases))
with open("data/onset_by_date_missing.pickle","rb") as f:
    onsets_per_date = pickle.load(f)
    print(onsets_per_date.shape)
    print(onsets_per_date[0])
    # date per line
    # days with report 
    
with open("data/onsets_by_date_diff.pickle","rb") as f:
    onsets_per_date_diff = pickle.load(f)
    print("Diff",onsets_per_date_diff.shape)


trace = None
with pm.Model() as model:

    # Known working parameter Set: 1,.1 1,.1 t1=10,t2=24 offset=7 --> 2.53+/-0.050 / 0.99+/-0.070 
    # 6,24 o7 --> 0.97+/-0.021 / 0.83+/-0.066
    # a1 = pm.Lognormal(name="initial_a1",mu=1,sigma=.1)
    # a2 = pm.HalfNormal(name="initial_a2",sigma=.3)
    # t2 = pm.Lognormal(name="initial_t2",mu=tt.log(20),sigma=.1)
    # imported_cases = GenInit(num_days_sim,a1,a2,t2=t2,offset=5)
    # pm.Deterministic("initial_t",imported_cases)
    
     
    initial_lambda = pr_d["pr_median_lambda_0"]  
    # initial_lambda = pm.Lognormal(
  #               name="lambda_0",
  #               mu=np.log(pr_d["pr_median_lambda_0"]),
  #               sigma=pr_d["pr_sigma_lambda_0"],
  #          )
#    flt = np.array([.2,.7,.7,1,.7,.7,.2],dtype=np.float64)
    # flt = np.array([1,.5,.2])
    flt = np.array([.1,1.,.1],dtype=np.float64)
    f_weekend = 1.
    f_weekend = pm.Lognormal(name="f_weekend",mu=tt.log(f_weekend),sigma=.1)
    lambda_t,week_mask = DailyRandomWalkWeekend("lambda_t",mi.length,initial_lambda,f_weekend,flt=flt,offset=mi.weekoffset)
    pm.Deterministic("lambda_t",lambda_t)
    
    initial_delay_ratio = pm.Uniform("delay_ratio_0",lower=.01,upper=.99)
#    delay_ratio_t = WeeklyRandomWalk("delay_ratio_t",num_days_sim,initial_delay_ratio,sigma=.01,flt=np.array([1,1,1,1,1,1,1],dtype=np.float64))
    delay_ratio_t = WeeklyRandomWalk("delay_ratio_t",mi.length,initial_delay_ratio,sigma=.01,flt=np.array([.5,1,1,1,1,1,.5],dtype=np.float64),offset=mi.weekoffset)
    
    delay_ratio_t = tt.clip(delay_ratio_t,.01,.99)
    pm.Deterministic("delay_ratio_t",delay_ratio_t)
    
    
    #
 #   s_import = 1.
 #   s_import = pm.Lognormal(name="median_imported_factor",mu=tt.log(1.),sigma=.05)
    
    # Core Function
#    f_unknown = pm.Lognormal(name="f_unknown",mu=tt.log(1.1),sigma=.1)
    median_infectious,sigma_infectious = 4.,.466
    median_infectious = pm.Lognormal(name = "median_infectious",mu = tt.log(median_infectious),sigma = .1)
    infected_t,infected_v,N_t = SEIR_model(86e6,imported_cases[:mi.length],lambda_t[:mi.length],median_infectious,sigma_infectious,l=32)
    pm.Deterministic("infected_t",infected_t)
    
    
    median_incubation,sigma_incubation = 4.,.466
    median_incubation = pm.Lognormal(name = "median_incubation",mu = tt.log(median_incubation),sigma = 0.1)
    onset_of_illness_t = DelayLognormal(infected_t,median=median_incubation,sigma=sigma_incubation,factor=1.,l=32)[:mi.length]
    pm.Deterministic("onset_of_illness_t",onset_of_illness_t)
    
#    f_wtrans = 0.15
    f_wtrans = pm.HalfCauchy(name="w_trans",beta=.3)
    reported_onset_of_illness_t = TransferWeekendReported(onset_of_illness_t,f_wtrans,week_mask)
    pm.Deterministic("reported_onset_of_illness_t",reported_onset_of_illness_t)
    
    delay_m1,delay_s1,delay_m2,delay_s2 = 8,.4,20,.5
    delay_m1 = pm.Lognormal(name="delay_m1",mu=tt.log(delay_m1),sigma=.2)
    delay_s1 = pm.HalfNormal(name="delay_s1",sigma=delay_s1)
#    delay_m2 = pm.Lognormal(name="delay_m2",mu=tt.log(delay_m2),sigma=.5)
    delay_s2 = pm.HalfNormal(name="delay_s2",sigma=delay_s2)

    
    per_day_curves_t = reportDelayDistFunc(reported_onset_of_illness_t[:mi.length],
                                                            delay_m1,delay_s1,
                                                            delay_m2,delay_s2, delay_ratio_t,n=128)
    
    
    publishing_start = datetime.date(2020,3,4)  # Startdate of epicurves
    onset_start = datetime.date(2020,2,16)    # Startdate of known onsets
    ofront = (onset_start-mi.startdate).days        # 4
    pfront = (publishing_start-mi.startdate).days   # 22
    
    per_day_curves_dt = DelayDaily(per_day_curves_t[:mi.length,:mi.length],3,week_mask)
    per_day_curves_s = tt.cumsum(per_day_curves_t,axis=1)
    per_day_curves_tdt = per_day_curves_dt[ofront:ofront+onsets_per_date_diff.shape[0],pfront+1:pfront+onsets_per_date_diff.shape[1]+1] # 
    
    pm.Deterministic("per_day_curve_t",per_day_curves_t)
    pm.Deterministic("per_day_curve_dt",per_day_curves_dt)
    pm.Deterministic("per_day_curve_s",per_day_curves_s)
    
    
    
    reported_t = tt.sum(per_day_curves_dt,axis=0)
    pm.Deterministic("reported_t",reported_t)
    
    
    if False:
        sigma_onset_obs = pm.HalfCauchy( name="sigma_onset_obs", beta=50 )
#        sigma_delayed_obs = pm.Lognormal(name="sigma_delayed_obs",mu=tt.log(10),sigma=.8)
        pm.StudentT(
                name="new_day_curve_studentT",
                nu=4,
                mu=per_day_curves_s[ofront:ofront+onsets_per_date.shape[0],pfront+onsets_per_date.shape[1]-1],
#                sigma=tt.abs_(per_day_curves_s[4:-2,-2] + tt.alloc(1,per_day_curves_s.shape[1])) ** 0.5 * sigma_onset_obs,  # +1 and tt.abs to avoid nans
        
                sigma=tt.abs_(per_day_curves_s[ofront:ofront+onsets_per_date.shape[0],pfront+onsets_per_date.shape[1]-1] + 1) ** 0.5 * sigma_onset_obs,  # +1 and tt.abs to avoid nans
                observed=onsets_per_date[:,-1]
           )
    if True:
        # Diff-Obs
        sigma_onset_diff_obs = pm.HalfCauchy( name="sigma_onset_diff_obs", beta=30 )
        pm.StudentT(
                name="new_day_curve_diff_studentT",
                nu=4,
                mu=per_day_curves_tdt,
                sigma=tt.abs_(per_day_curves_tdt + tt.alloc(1,per_day_curves_tdt.shape[1])) ** 0.5 * sigma_onset_diff_obs,  # +1 and tt.abs to avoid nans
                observed=onsets_per_date_diff
           )
    
    # if False:
    #     sigma_reported_obs = pm.HalfCauchy( name="sigma_reorted_obs",beta=10 )
    #     pm.StudentT(
    #             name="new_reported_cases_studentT",
    #             nu=4,
    #             mu=onset_of_illness_t[5:-12],
    #             sigma=tt.abs_(onset_of_illness_t[5:-11] + 1) ** 0.5 * sigma_reported_obs,  # +1 and tt.abs to avoid nans
    #             observed=reported
    #         )
    
#     if False:
#         m_hosp,s_hosp,f_hosp = 14,1.5,.22
#         f_hosp = pm.Lognormal(name="f_hospital",mu=tt.log(f_hosp),sigma=.01)
#         m_hosp = pm.Normal(name="m_hospital",mu=m_hosp,sigma=1.)
# #        s_hosp = pm.HalfNormal(name="s_hospital",sigma=s_hosp) # Greatly slows down the
#         hospital_t = DelayLognormal(infected_t,median=m_hosp,sigma=s_hosp,factor=f_hosp,l=36)[hospital_begin:num_days_sim]
#         sigma_obs_hospital = pm.HalfCauchy( name="sigma_obs_hospital", beta=10 )
#         pm.StudentT(
#                 name="new_hospital_studentT",
#                 nu=4,
#                 mu=hospital_t,
#                 sigma=tt.abs_(hospital_t+.5) ** 0.5*sigma_obs_hospital,
#                 observed=hospital[:(num_days_sim-hospital_begin)]
#             )
#         pm.Deterministic("hosp_t",hospital_t)
#
    if False:
        m_death,s_death,f_death = 25,.4,.04
        f_death = pm.Normal(name="f_death",mu=f_death,sigma=.01)
        m_death = pm.Normal(name="m_death",mu=m_death,sigma=1.)
      #  s_death = pm.Lognormal(name="s_death",mu=tt.log(.5),sigma=.1)
        dead_t = DelayLognormal(infected_t,median=m_death,sigma=s_death,factor=f_death,l=42)[:num_days_sim]
        sigma_obs_dead = pm.HalfCauchy( name="sigma_obs_dead", beta=10 )
        pm.StudentT(
                name="new_deaths_studentT",
                nu=4,
                mu=dead_t,
                sigma=tt.abs_(dead_t+.5) ** 0.5*sigma_obs_dead,
                observed=deaths[:num_days_sim]
            )
        pm.Deterministic("dead_t",dead_t)
#
#     if False:
#         m_ICU_death,s_ICU_death,f_ICU_death = 10,.5,.1
#         f_ICU_death = pm.Normal(name="f_ICU_death",mu=f_death,sigma=.01)
#         m_ICU_death = pm.Normal(name="m_ICU_death",mu=m_death,sigma=1.5)
#         #  s_death = pm.Lognormal(name="s_death",mu=tt.log(.5),sigma=.1)
#         ICU_dead_t = DelayLognormal(hospital_t,median=m_ICU_death,sigma=s_ICU_death,factor=f_ICU_death,l=24)[icu_deaths_begin:num_days_sim]
#         sigma_obs_ICU_dead = pm.HalfCauchy( name="sigma_obs_ICU_dead", beta=10 )
#         pm.StudentT(
#                 name="new_ICU_deaths_studentT",
#                 nu=4,
#                 mu=ICU_dead_t,
#                 sigma=tt.abs_(ICU_dead_t+.5) ** 0.5*sigma_obs_ICU_dead,
#                 observed=icu_deaths[:(num_days_sim-icu_deaths_begin)]
#             )
#         pm.Deterministic("ICU_dead_t",ICU_dead_t)

#    model.profile(model.logpt).summary()
#    model.profile(gradient(model.logpt, model.vars)).summary()

#init='advi'
#'advi+adapt_diag'
#init='advi+adapt_diag' 
#trace = pm.sample(model=model,init='advi+adapt_diag',draws=300,cores=2,chains=2,tune=300,n_init=1000000,max_treedepth=12)
trace = pm.sample(model=model,draws=400,cores=2,chains=2,tune=500,max_treedepth=11)

if trace != None:
    d = datetime.datetime.now()
    dn = "%02d%02d%02d"%(d.hour,d.minute,d.second)
    fn = "traces/trace"+dn+".dat"

    with open(fn,"wb+") as f:
        pickle.dump(trace,f)
        pickle.dump(mi,f)
        pickle.dump(ds,f)
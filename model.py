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

import matplotlib.pyplot as plt

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
#theano.config.floatX = "float64"
#theano.config.compute_test_value = "ignore"
#theano.config.exception_verbosity="high"
theano.config.optimizer='fast_run'  
#theano.config.mode='DebugMode'  

def tt_lognormal(x, mu, sigma):
# Limit to prevent NANs
    x = tt.clip(x,1e-9,1e12)
    sigma = tt.clip(sigma,1e-9,1e12)
    mu = tt.clip(mu,1e-9,1e12)
    
    distr = 1/x * tt.exp( -( (tt.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return distr / (tt.sum(distr, axis=0) + 1e-12)

def reportDelayDist(x, mu1, sig1, mu2, sig2, r ):
    """ builds the reporting Delay distribution from two lognormal distributions"""
#    x = tt.clip(x,1e-12,1e12) # Checks to improve stability and resilience against NANs
    r = tt.clip(r,0,1)
    xm = tt.alloc(x,r.shape[0],x.shape[0])
    d1 = tt_lognormal(x,tt.log(mu1),sig1) * xm
    d2 = tt_lognormal(x,tt.log(mu2),sig2) * (1-xm)
    ds = d1+d2
    return ds / (tt.sum(ds,axis=0) + 1e-12)
    
def DelayedReporting(cases,mu1,sig1,mu2,sig2,r,n=64):
    cases = tt.cast(cases,'float64')
    x = tt.arange(1,n+1)
    dist = reportDelayDist(x,mu1,sig1,mu2,sig2,r)
    
    mc = tt.nlinalg.alloc_diag(cases)
  #  dist0 = tt.alloc(0.,1,dist.shape[0])
  #  dist0d = tt.cast(dist0,'float64')
    distm = tt.set_subtensor(dist0d[0,:],dist)

    mcc = tt.signal.conv.conv2d(mc,distm,border_mode='full')
    ds = tt.sum(mcc,axis=1)
    
    return ds,mcc,dist

def conv_offset(inp,filt,amplitude=1,offset=0):
    amplitude = tt.clip(amplitude,1e-12,1e9) # Limit to prevent NANs
    
    zero = tt.zeros_like(inp)
    a0 = tt.concatenate((inp,zero,),0)
    a0r = tt.roll(a0,offset,0)
    a0rp = tt.set_subtensor(a0r[:offset],0.) * amplitude
    
    a0rp3d = tt.alloc(0.,1,a0rp.shape[0],1 )
    a0rp = tt.set_subtensor(a0rp3d[0,:,0],a0rp)
    filt3d = tt.alloc(0.,1,filt.shape[0],1 )
    filt = tt.set_subtensor(filt3d[0,:,0],filt)
    return tt_conv.conv2d(a0rp,filt,None,None,border_mode='full').flatten()
    
def est_deaths(infected_t,median,sigma,factor,l=40,offset=0):
    beta = tt_lognormal(tt.arange(l*2),tt.log(median),sigma)
    return conv_offset(infected_t,beta,factor,offset)

def WeeklyRandomWalk(name,n,initial,flt=np.array([.05,.1,.7,.1,.05]),sigma=.05):
    
    delay_list_length = n//7+1
    rw_list = []
    rw_list.append(initial)
    sigma_random_walk = pm.HalfNormal(name=name+"_sigma_random_walk", sigma=sigma)
    delay_ratio_random_walk = pm.distributions.timeseries.GaussianRandomWalk(
                              name=name+"_random_walk",mu=0,
                              sigma=sigma,shape=delay_list_length,
                              init=pm.Normal.dist(sigma_random_walk),
                        )
    flt = flt / tt.sum(flt)
    val = delay_ratio_random_walk
    lval = tt.alloc(0.,val.shape[0]+2)
    lval = tt.cast(lval,"float64")
    lval = tt.set_subtensor(lval[1:-1],val)
    lval = tt.set_subtensor(lval[0],val[0])
    lval = tt.set_subtensor(lval[-1],val[-1])
    
    m = tt.alloc(lval,7,lval.shape[0])
    mf = tt.flatten(m.T,ndim=1)
    
    mf2 = tt.alloc(mf,1,mf.shape[0])
    kern2 = tt.alloc(flt,1,flt.shape[0])
    
    r = tt.signal.conv.conv2d(mf2,kern2,border_mode='full')
    rs = r[0,(7+flt.shape[0]//2):(7*val.shape[0]+7+flt.shape[0]//2)][:n]
    
    return rw_list[0]+rs

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
pr_d["pr_median_lambda_0"] = 2.5
pr_d["pr_sigma_lambda_0"] = .5

# Data
epi_curve = [0,0,0,0]+[7, 11, 8, 10, 34, 22, 26, 46, 65, 103, 147, 155, 195, 177, 193, 268, 255, 349, 363, 575, 744, 989, 1502, 1924, 2394, 2651, 3240, 3261, 3385, 4518, 3753, 3900, 3343, 3923, 3202, 2636, 3777, 2803, 3045, 2707, 2764, 2619, 2113, 2970, 2186, 2575, 2298, 2356, 1861, 1668, 2135, 1810, 1659, 1612, 1311, 1134, 1136, 1095, 1130, 1061, 933, 894, 795, 667, 834, 647, 614, 589, 515, 415, 351, 430, 295, 211, 160, 45, 18, 10, 2][:-3]
epi_curve = np.asarray(epi_curve,dtype=np.float64)
epi_curve = ma.masked_less_equal(epi_curve,0.,copy=True)

initial_obs_rki_sit = [x*.5 for x in [9.188,9.188,4.594,4.594,22.971,13.783,27.565,27.565,27.565,53.507,32.821]]+[30.,16.,12.,20.,30.,40,40]+[63.072,106.910,163.956,231.851,306.731,383.947,458.793,527.101,585.620,632.178,665.686,686.005,693.759,690.122,676.616,654.935,626.805,593.878,557.666,519.498,480.503,441.606,403.541,366.868,331.992,299.187,268.619,240.364,214.431,190.776,169.313,149.934,132.509,116.901,102.965,90.561,79.549,69.796,61.177,53.573,46.877,40.989,35.819,31.284,27.311,23.833,20.792,18.134,15.812,13.785,12.017,10.475,9.130,7.958,6.937,6.047,5.272,4.596,4.008,3.496,3.049,2.661,2.322,2.027,1.770]+[0]*7

imported_cases = np.asarray(initial_obs_rki_sit[:len(epi_curve)],dtype=np.float64)
#imported_cases = np.asarray([5,4,3,2,1]+[0]*(len(epi_curve)-6-6),dtype=np.float64)

hospital = [-1]*17+[-1]*20+[379,352,529,581,661,432,566,719,759,834,839,748,747,500,676,751,791,789,604,468,425,415,644,705,645,660,428,310,405,569,500,490,410,286,160,296,360,379,470,192,132,130]+[-1]*5
hospital_begin = 37
hospital = ma.masked_less(hospital,0.,copy=True)[hospital_begin:]

icu_deaths = [-1]*17+[-1]*36+[93,154,0,131,88,66,228,255,305,123,50,59,108,91,115,39,22,29,24,68,69,52,40,46,85]
icu_deaths_begin = 17+30
icu_deaths = ma.masked_less(icu_deaths,0.,copy=True)[icu_deaths_begin:]

deaths = [-1]*(17+4)+[-1,-1,-1,-1,-1,-1,0,1,2,0,3,4,0,0,0,8,11,16,8,31,28,35,49,55,72,64,66,128,149,140,145,141,184,92,173,254,246,266,171,129,126,170,285,315,299,242,184,110,194,281,215,227,179,140,110,163,202,173,193,94,74,43,139]
deaths = np.asarray(deaths,dtype=np.float64)
deaths = ma.masked_less(deaths,0.,copy=True)

lambda_t = np.array( [2.045,2.050,2.050,2.054,2.056,2.059,2.064,2.067,2.071,2.076,2.080,2.083,2.082,2.080,2.074,2.071,2.068,2.065,2.060,2.052,2.043,2.033,2.019,1.996,1.953,1.895,1.823,1.731,1.628,1.527,1.406,1.285,1.163,1.070,0.968,0.897,0.867,0.806,0.780,0.755,0.752,0.746,0.752,0.783,0.772,0.793,0.790,0.787,0.754,0.740,0.759,0.748,0.731,0.714,0.685,0.663,0.662,0.667,0.682,0.682,0.680,0.679,0.669,0.665,0.677,0.670,0.672,0.677,0.680,0.683,0.694,0.718,0.750,0.774,0.789,0.793]+[.8]*4 ,dtype=np.float32)

print("Len of inputs",len(deaths),len(epi_curve),len(imported_cases),len(hospital),print(icu_deaths))


with open("data/onset_by_date.pickle","rb") as f:
    onsets_per_date = pickle.load(f)
    print(onsets_per_date.shape)
    # date per line
    # days with report 

trace = None
with pm.Model() as model:
    num_days_sim = min(len(epi_curve),len(imported_cases))
    
    # if False:
   #      lambda_list = []
   #      lambda_list.append(
   #          pm.Lognormal(
   #              name="lambda_0",
   #              mu=np.log(pr_d["pr_median_lambda_0"]),
   #              sigma=pr_d["pr_sigma_lambda_0"],
   #          )
   #      )
   #      # build the time-dependent spreading rate
   #      if pr_d.get("pr_sigma_random_walk",0) != 0:
   #          sigma_random_walk = pm.HalfNormal(
   #              name="sigma_random_walk", sigma=pr_d["pr_sigma_random_walk"]
   #          )
   #          lambda_t_random_walk = pm.distributions.timeseries.GaussianRandomWalk(
   #              name="lambda_t_random_walk",
   #              mu=0,
   #              sigma=sigma_random_walk,
   #              shape=num_days_sim,
   #              init=pm.Normal.dist(sigma=pr_d["pr_sigma_random_walk"]),
   #          )
   #          lambda_base = lambda_t_random_walk + lambda_list[0]
   #      else:
   #          lambda_base = lambda_list[0] * tt.ones(num_days_sim)
   #
   #      lambda_t_list = [lambda_base]
        # Transition-points
        # lambda_step_before = lambda_list[0]
        # for tr_begin, transient_len, lambda_step in zip(
        #     tr_begin_list, tr_len_list, lambda_list[1:]
        # ):
        #     lambda_t = mh.smooth_step_function(
        #         start_val=0,
        #         end_val=1,
        #         t_begin=tr_begin,
        #         t_end=tr_begin + transient_len,
        #         t_total=num_days_sim,
        #     ) * (lambda_step - lambda_step_before)
        #     lambda_step_before = lambda_step
        #     lambda_t_list.append(lambda_t)
    
     #   lambda_t = sum(lambda_t_list)
        
        
    initial_lambda = pm.Lognormal(
                name="lambda_0",
                mu=np.log(pr_d["pr_median_lambda_0"]),
                sigma=pr_d["pr_sigma_lambda_0"],
            )
    lambda_t = WeeklyRandomWalk("lambda_t",num_days_sim,initial_lambda)
    pm.Deterministic("lambda_t",lambda_t)
    
    initial_delay_ratio = pm.Uniform("delay_ratio_0",lower=.01,upper=.99)
    delay_ratio_t = WeeklyRandomWalk("delay_ratio_t",64,initial_delay_ratio)
    pm.Deterministic("delay_ratio_t",delay_ratio_t)
    
    median_incubation = 4.
    median_incubation = pm.Lognormal(
         name = "median_incubation",
         mu = tt.log(median_incubation),
         sigma = 0.04
    )
    
    #
 #   s_import = 1.
 #   s_import = pm.Lognormal(name="median_imported_factor",mu=tt.log(1.),sigma=.05)
    
    # Core Function
    
    infected_t,infected_v,N_t = SEIR_model(86e6,imported_cases[:num_days_sim],lambda_t[:num_days_sim],median_incubation,.466,l=32)
#    sigma_obs = pm.HalfCauchy( name="sigma_obs", beta=10 )
    pm.Deterministic("infected_t",infected_t)
    onset_of_illness_t = est_deaths(infected_t,median=(median_incubation),sigma=.7,factor=1,l=12)[:num_days_sim]
    pm.Deterministic("onset_of_illness_t",onset_of_illness_t)
    
    delay_m1,delay_s1,delay_m2,delay_s2 = 10,.2,25,.4
#    delay_ratio = pm.Uniform(name="delay_ratio",lower=0,upper=1)
    delay_m1 = pm.Lognormal(name="delay_m1",mu=tt.log(delay_m1),sigma=.2)
 #   delay_s1 = pm.HalfNormal(name="delay_s1",sigma=delay_s1)
    delay_m2 = pm.Lognormal(name="delay_m2",mu=tt.log(delay_m2),sigma=.5)
#    delay_s2 = pm.HalfNormal(name="delay_s2",sigma=delay_s2)
    epi_curve_t,per_day_curves,delay_distr = DelayedReporting(onset_of_illness_t[:num_days_sim],delay_m1,delay_s1,delay_m2,delay_s2,r=delay_ratio_t,n=64)
    pm.Deterministic("delayed_distr",delay_distr)
    pm.Deterministic("epi_curve_t",epi_curve_t)
    
    epi_curve_tc = epi_curve_t[4:num_days_sim+4]
    per_day_curves_t = per_day_curves[:num_days_sim,(num_days_sim-63):num_days_sim],
    per_day_curves_s = tt.cumsum(per_day_curves_t[0],axis=1)

    pm.Deterministic("per_day_curve_t",per_day_curves)
    pm.Deterministic("per_day_curve_s",per_day_curves_s)
    
    print(per_day_curves_s)
    print(onsets_per_date.shape)
    
#    sigma_delayed_obs = pm.HalfCauchy( name="sigma_delayed_obs", beta=5 )
    sigma_delayed_obs = pm.Lognormal(name="sigma_delayed_obs",mu=tt.log(10),sigma=.8)
    pm.StudentT(
            name="new_day_curve_studentT",
            nu=4,
            mu=per_day_curves_s,
            sigma=tt.abs_(per_day_curves_s + tt.alloc(1,63)) ** 0.5 * sigma_delayed_obs,  # +1 and tt.abs to avoid nans
            observed=onsets_per_date
       )
    
    # pm.StudentT(
    #         name="new_cases_studentT",
    #         nu=4,
    #         mu=epi_curve_tc,
    #         sigma=tt.abs_(epi_curve_tc + 1) ** 0.5 * sigma_obs,  # +1 and tt.abs to avoid nans
    #         observed=epi_curve[:76]
    #     )
    
#     if False:
#         m_hosp,s_hosp,f_hosp = 14,1.5,.22
#         f_hosp = pm.Lognormal(name="f_hospital",mu=tt.log(f_hosp),sigma=.01)
#         m_hosp = pm.Normal(name="m_hospital",mu=m_hosp,sigma=1.)
# #        s_hosp = pm.HalfNormal(name="s_hospital",sigma=s_hosp) # Greatly slows down the
#         hospital_t = est_deaths(infected_t,median=m_hosp,sigma=s_hosp,factor=f_hosp,l=36)[hospital_begin:num_days_sim]
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
    if True:
        m_death,s_death,f_death = 25,.4,.04
        f_death = pm.Normal(name="f_death",mu=f_death,sigma=.01)
        m_death = pm.Normal(name="m_death",mu=m_death,sigma=1.)
      #  s_death = pm.Lognormal(name="s_death",mu=tt.log(.5),sigma=.1)
        dead_t = est_deaths(infected_t,median=m_death,sigma=s_death,factor=f_death,l=42)[:num_days_sim]
        sigma_obs_dead = pm.HalfCauchy( name="sigma_obs_dead", beta=100 )
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
#         ICU_dead_t = est_deaths(hospital_t,median=m_ICU_death,sigma=s_ICU_death,factor=f_ICU_death,l=24)[icu_deaths_begin:num_days_sim]
#         sigma_obs_ICU_dead = pm.HalfCauchy( name="sigma_obs_ICU_dead", beta=10 )
#         pm.StudentT(
#                 name="new_ICU_deaths_studentT",
#                 nu=4,
#                 mu=ICU_dead_t,
#                 sigma=tt.abs_(ICU_dead_t+.5) ** 0.5*sigma_obs_ICU_dead,
#                 observed=icu_deaths[:(num_days_sim-icu_deaths_begin)]
#             )
#         pm.Deterministic("ICU_dead_t",ICU_dead_t)


#init='advi'
trace = pm.sample(model=model,init='advi+adapt_diag' , draws=50,cores=2,chains=2,tune=50)



if trace != None:
    d = datetime.datetime.now()
    ds = "%02d%02d%02d"%(d.hour,d.minute,d.second)
    fn = "traces/trace"+ds+".dat"

    with open(fn,"wb+") as f:
        pickle.dump(trace,f)
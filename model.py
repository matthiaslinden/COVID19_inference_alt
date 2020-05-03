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
    
    distr = 1/(x+1e-9) * tt.exp( -( (tt.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return distr / (tt.sum(distr, axis=0) + 1e-12)
	
def conv_offset(inp,filt,amplitude=1,offset=0):
    zero = tt.zeros_like(inp)
    a0 = tt.concatenate((inp,zero,),0)
    a0r = tt.roll(a0,offset,0)
    a0rp = tt.set_subtensor(a0r[:offset],0.) * amplitude

    amplitude = tt.clip(amplitude,1e-12,1e9) # Limit to prevent NANs
    
    a0rp3d = tt.alloc(0.,1,a0rp.shape[0],1 )
    a0rp = tt.set_subtensor(a0rp3d[0,:,0],a0rp)
    filt3d = tt.alloc(0.,1,filt.shape[0],1 )
    filt = tt.set_subtensor(filt3d[0,:,0],filt)
    return tt_conv.conv2d(a0rp,filt,None,None,border_mode='full').flatten()
    
def est_deaths(infected_t,median,sigma,factor,l=40,offset=0):
    beta = tt_lognormal(tt.arange(l*2),tt.log(median),sigma)
    return conv_offset(infected_t,beta,factor,offset)


def SIR_model(N, imported_t,lambda_t, median_incubation,sigma_incubation,l=32):
    N = tt.cast(N,'float64')
    beta = tt_lognormal(tt.arange(l), tt.log(median_incubation), sigma_incubation)
    
    # Dirty hack to prevent nan
 #   beta = tt.alloc(0,l)
  #  beta = tt.set_subtensor(beta[tt.clip(tt.cast(median_incubation,'int32'),1,l-2)],1)
     
    lambda_t = tt.as_tensor_variable(lambda_t)
    imported_t = tt.as_tensor_variable(imported_t)

    def new_day(lambda_at_t,imported_at_t,infected,N_at_t,beta,N):
        f = N_at_t / N
    #    f = 1
        new = imported_at_t + theano.dot(infected,beta) * lambda_at_t * f
     #   new = tt.clip(new,0,N)
        infected = tt.roll(infected,1,0)
        infected = tt.set_subtensor(infected[:1],new,inplace=False)
        N_at_t = tt.clip(N_at_t-new,0,N_at_t)
        N_at_t = N_at_t-new
        return new,infected,N_at_t
    
    outputs_info = [None,np.zeros(l),N]
    infected_t,updates = theano.scan(fn=new_day,
                                     sequences=[lambda_t,imported_t],
                                     outputs_info=outputs_info,
                                     non_sequences=[beta,N],
                                     profile=True)
                                     
    return infected_t


pr_d = {}
#priors_dict[""] = 
pr_d["pr_beta_sigma_obs"] = 10
pr_d["pr_mean_median_incubation"] = 4.
pr_d["sigma_incubation"] = 0.418
pr_d["pr_sigma_random_walk"] = 0.01 #0.05
pr_d["pr_median_lambda_0"] = 2.2
pr_d["pr_sigma_lambda_0"] = .5

# Data
epi_curve = [7, 10, 8, 10, 34, 22, 26, 46, 64, 103, 147, 155, 196, 175, 193, 268, 254, 347, 363, 573, 745, 988, 1497, 1920, 2393, 2641, 3227, 3250, 3376, 4499, 3740, 3882, 3328, 3905, 3187, 2625, 3755, 2791, 3031, 2688, 2750, 2608, 2101, 2946, 2181, 2556, 2282, 2349, 1844, 1651, 2123, 1800, 1646, 1594, 1305, 1122, 1128, 1081, 1122, 1039, 911, 874, 766, 660, 812, 627, 589, 555, 468, 357, 304, 351, 192, 118, 68, 4, 1]
epi_curve = np.asarray(epi_curve[:-5]+[-1]*10,dtype=np.float64)
epi_curve = ma.masked_less_equal(epi_curve,0.,copy=True)

initial_obs_rki_sit = [9.188,9.188,4.594,4.594,22.971,13.783,27.565,27.565,27.565,53.507,32.821]+[x*.8 for x in [30,17,16,36]+[63.072,106.910,163.956,231.851,306.731,383.947,458.793,527.101,585.620,632.178,665.686,686.005,693.759,690.122,676.616,654.935,626.805,593.878,557.666,519.498,480.503,441.606,403.541,366.868,331.992,299.187,268.619,240.364,214.431,190.776,169.313,149.934,132.509,116.901,102.965,90.561,79.549,69.796,61.177,53.573,46.877,40.989,35.819,31.284,27.311,23.833,20.792,18.134,15.812,13.785,12.017,10.475,9.130,7.958,6.937,6.047,5.272,4.596,4.008,3.496,3.049]]#,2.661,2.322]#,2.027,1.770]
imported_cases = np.asarray(initial_obs_rki_sit[:len(epi_curve)],dtype=np.float64)

hospital = [-1]*17+[-1]*20+[379,352,529,581,661,432,566,719,759,834,839,748,747,500,676,751,791,789,604,468,425,415,644,705,645,660,428,310,405,569,500,490,410,286,160,296,360,379,470,192]+[-1]*5
hospital_begin = 37
hospital = ma.masked_less(hospital,0.,copy=True)[hospital_begin:]

icu_deaths = [-1]*17+[-1]*36+[93,154,0,131,88,66,228,255,305,123,50,59,108,91,115,39,22,29,24,68,69,52,40,46]
icu_deaths_begin = 17+30
icu_deaths = ma.masked_less(icu_deaths,0.,copy=True)[icu_deaths_begin:]

deaths = [-1]*17+[-1,-1,-1,-1,-1,-1,0,1,2,0,3,4,0,0,0,8,11,16,8,31,28,35,49,55,72,64,66,128,149,140,145,141,184,92,173,254,246,266,171,129,126,170,285,315,299,242,184,110,194,281,215,227,179,140,110,163,202,173,193]
deaths = np.asarray(deaths,dtype=np.float64)
deaths = ma.masked_less(deaths,0.,copy=True)

print("Len of inputs",len(deaths),len(epi_curve),len(imported_cases),len(hospital),print(icu_deaths))

trace = None
with pm.Model() as model:
    num_days_sim = min(len(epi_curve),len(imported_cases))
    
    lambda_list = []
    lambda_list.append(
        pm.Lognormal(
            name="lambda_0",
            mu=np.log(pr_d["pr_median_lambda_0"]),
            sigma=pr_d["pr_sigma_lambda_0"],
        )
    )
    # build the time-dependent spreading rate
    if pr_d.get("pr_sigma_random_walk",0) != 0:
        sigma_random_walk = pm.HalfNormal(
            name="sigma_random_walk", sigma=pr_d["pr_sigma_random_walk"]
        )
        lambda_t_random_walk = pm.distributions.timeseries.GaussianRandomWalk(
            name="lambda_t_random_walk",
            mu=0,
            sigma=sigma_random_walk,
            shape=num_days_sim,
            init=pm.Normal.dist(sigma=pr_d["pr_sigma_random_walk"]),
        )
        lambda_base = lambda_t_random_walk + lambda_list[0]
    else:
        lambda_base = lambda_list[0] * tt.ones(num_days_sim)

    lambda_t_list = [lambda_base]
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

    lambda_t = sum(lambda_t_list)
    
    median_incubation = 4.
    median_incubation = pm.Lognormal(
         name = "median_incubation",
         mu = tt.log(median_incubation),
         sigma = 0.04
    )
    pm.Deterministic("lambda_t",lambda_t)
    #
 #   s_import = 1.
 #   s_import = pm.Lognormal(name="median_imported_factor",mu=tt.log(1.),sigma=.05)
    
    # Core Function
#    scaled_imported_cases = imported_cases*s_import;
    infected_t,infected_v,N_t = SIR_model(86e6,imported_cases[:num_days_sim],lambda_t[:num_days_sim],median_incubation,.466,l=20)
    sigma_obs = pm.HalfCauchy( name="sigma_obs", beta=10 )
    pm.StudentT(
            name="new_cases_studentT",
            nu=4,
            mu=infected_t,
            sigma=tt.abs_(infected_t + 1) ** 0.5 * sigma_obs,  # +1 and tt.abs to avoid nans
            observed=epi_curve[:num_days_sim]
        )
    pm.Deterministic("infected_t",infected_t)
    
    m_hosp,s_hosp,f_hosp = 14,1.5,.22
    f_hosp = pm.Lognormal(name="f_hospital",mu=tt.log(f_hosp),sigma=.01)
    m_hosp = pm.Normal(name="m_hospital",mu=m_hosp,sigma=1.)
    s_hosp = pm.HalfNormal(name="s_hospital",sigma=s_hosp) # Greatly slows down the 
    hospital_t = est_deaths(infected_t,median=m_hosp,sigma=s_hosp,factor=f_hosp,l=36)[hospital_begin:num_days_sim]
    sigma_obs_hospital = pm.HalfCauchy( name="sigma_obs_hospital", beta=10 )
    pm.StudentT(
            name="new_hospital_studentT",
            nu=4,
            mu=hospital_t,
            sigma=tt.abs_(hospital_t+.5) ** 0.5*sigma_obs_hospital,
            observed=hospital[:(num_days_sim-hospital_begin)]
        )
    pm.Deterministic("hosp_t",hospital_t)
    
    m_death,s_death,f_death = 25,.4,.04
    f_death = pm.Normal(name="f_death",mu=f_death,sigma=.01)
    m_death = pm.Normal(name="m_death",mu=m_death,sigma=1.5)
  #  s_death = pm.Lognormal(name="s_death",mu=tt.log(.5),sigma=.1)
    dead_t = est_deaths(infected_t,median=m_death,sigma=s_death,factor=f_death,l=42)[:num_days_sim]
    sigma_obs_dead = pm.HalfCauchy( name="sigma_obs_dead", beta=10 )
    pm.StudentT(
            name="new_deaths_studentT",
            nu=4,
            mu=dead_t,
            sigma=tt.abs_(dead_t+.5) ** 0.5*sigma_obs_dead,
            observed=deaths[:num_days_sim]
        )
    pm.Deterministic("dead_t",dead_t)
        
    if False:
        m_ICU_death,s_ICU_death,f_ICU_death = 10,.5,.1
        f_ICU_death = pm.Normal(name="f_ICU_death",mu=f_death,sigma=.01)
        m_ICU_death = pm.Normal(name="m_ICU_death",mu=m_death,sigma=1.5)
        #  s_death = pm.Lognormal(name="s_death",mu=tt.log(.5),sigma=.1)
        ICU_dead_t = est_deaths(hospital_t,median=m_ICU_death,sigma=s_ICU_death,factor=f_ICU_death,l=24)[icu_deaths_begin:num_days_sim]
        sigma_obs_ICU_dead = pm.HalfCauchy( name="sigma_obs_ICU_dead", beta=10 )
        pm.StudentT(
                name="new_ICU_deaths_studentT",
                nu=4,
                mu=ICU_dead_t,
                sigma=tt.abs_(ICU_dead_t+.5) ** 0.5*sigma_obs_ICU_dead,
                observed=icu_deaths[:(num_days_sim-icu_deaths_begin)]
            )
        pm.Deterministic("ICU_dead_t",ICU_dead_t)


#init='advi'
trace = pm.sample(model=model,init="advi" , draws=200,cores=2,chains=2,tune=300)



if trace != None:
    d = datetime.datetime.now()
    ds = "%02d%02d%02d"%(d.hour,d.minute,d.second)
    fn = "trace"+ds+".dat"

    with open(fn,"wb+") as f:
        pickle.dump(trace,f)
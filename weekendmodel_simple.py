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
from ModelInfo import Modelinfo,Datasets

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
theano.config.floatX = "float64"
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

def GenInit(l,a1,a2,t1=10,t2=27,offset=8):
    x = tt.arange(l)
    d1 = tt_lognormal(x,tt.log(t1),.8)*2350 #.4 / 23500
    d2 = tt_lognormal(x,tt.log(t2),.25)*12500
    
    
    din = d1*a1 + d2*a2
    din = tt.roll(din,-offset)
    din = tt.set_subtensor(din[-offset:],0.)
    return din
    
    

def reportDelayDistFunc(cases,mu1,sig1,mu2,sig2,r,n):
    m1 = tt.cast(mu1,'float64')
    s1 = tt.cast(sig1,'float64')
    m2 = tt.cast(mu2,'float64')
    s2 = tt.cast(sig2,'float64')
    sr = tt.cast(r,'float64')
    n = tt.cast(n,'int64')
    x = tt.arange(1,n+1)
    
    # Prepare the Distributions
    sr = tt.clip(r,1e-12,1-1e-12)
    d1 = tt_lognormal(x,tt.log(m1),s1)
    d2 = tt_lognormal(x,tt.log(m2),s2)
    
    print(d1,d2)

    d1 = tt.alloc(d1,1,d1.shape[0])
    d2 = tt.alloc(d2,1,d2.shape[0])
    # Prepare cases as diagonal of matrix
    cin = tt.cast(cases,'float64')
    c2d = tt.nlinalg.alloc_diag(cin)
    # Create a Vector
    
    cf1 = tt.signal.conv.conv2d(c2d,d1,border_mode='full')
    cf2 = tt.signal.conv.conv2d(c2d,d2,border_mode='full')
    
    cfo = (sr*cf1.T + (tt.ones_like(sr)-sr)*cf2.T).T
    reported = tt.cumsum(cfo,axis=1)
    return reported,cfo#,dists

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
    
def est_deaths(infected_t,median,sigma,factor,l=40):
    l = tt.cast(l,'int64')
    beta = tt_lognormal(tt.arange(l*2),tt.log(median),sigma)
    return conv_offset(infected_t,beta,factor)
    


def WeeklyRandomWalk(name,n,initial,flt=np.array([.05,.1,.7,.1,.05],dtype=np.float64),sigma=.05,offset=0):
    additional_week = np.array([0,1,1,1,1,1,1])[offset%7]
    offset = tt.cast(offset,"int64")
    delay_list_length = n//7+additional_week
    rw_list = []
    rw_list.append(initial)
    sigma_random_walk = pm.HalfNormal(name=name+"_sigma_random_walk", sigma=sigma)
    delay_ratio_random_walk = pm.distributions.timeseries.GaussianRandomWalk(
                              name=name+"_random_walk",mu=0,
                              sigma=sigma_random_walk,shape=delay_list_length,
                              init=pm.Normal.dist(sigma=sigma),
                        )
    flt = tt.cast(flt,np.float64)
    flt = flt / tt.sum(flt)
    val = delay_ratio_random_walk
  
    lval = tt.alloc(0.,val.shape[0]+4)
    lval = tt.cast(lval,"float64")
    lval = tt.set_subtensor(lval[2:-2],val)
    lval = tt.set_subtensor(lval[:2],val[0])
    lval = tt.set_subtensor(lval[-2:],val[-1]) # extend the 

    
    m = tt.alloc(lval,7,lval.shape[0])
    mf = tt.flatten(m.T,ndim=1)
    mf = tt.roll(mf,offset)
    
    mf2 = tt.alloc(mf,1,mf.shape[0])
    kern2 = tt.alloc(flt,1,flt.shape[0])
    
    r = tt.signal.conv.conv2d(mf2,kern2,border_mode='full')
    r = tt.roll(r[0],offset)
    rs = r[(14+flt.shape[0]//2):(7*val.shape[0]+14+flt.shape[0]//2)][:n]
   
    return rw_list[0]+rs
    
def WeeklyRandomWalkWeekend(name,n,initial,wfactor,flt=np.array([.05,.1,.7,.1,.05],dtype=np.float64),sigma=.05,offset=0):
    additional_week = np.array([0,1,1,1,1,1,1])[offset%7]   # if firstday == monday, no additional week is needed
    offset = tt.cast(offset,"int64")
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
        new = imported_at_t + theano.dot(infected,beta) * lambda_at_t * f
        new = tt.clip(new,0,N)
     
        infected = tt.roll(infected,1,0)
        infected = tt.set_subtensor(infected[:1],new,inplace=False)
        E_t = tt.clip(E_t-new,0,E_t)
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
pr_d["pr_sigma_random_walk"] = 0.05
pr_d["pr_median_lambda_0"] = 4
pr_d["pr_sigma_lambda_0"] = .5

# Data
front = [-1]*4
epi_curve = front+[7, 9, 8, 10, 34, 19, 26, 46, 65, 102, 148, 158, 198, 179, 198, 275, 261, 356, 370, 590, 753, 1015, 1534, 1964, 2464, 2702, 3305, 3328, 3464, 4589, 3833, 3969, 3414, 4002, 3253, 2702, 3860, 2867, 3109, 2770, 2853, 2668, 2153, 3037, 2241, 2645, 2371, 2411, 1904, 1704, 2190, 1856, 1720, 1649, 1367, 1166, 1190, 1136, 1169, 1113, 976, 942, 829, 711, 898, 700, 675, 670, 605, 496, 436, 595, 417, 388, 398, 284, 260, 269, 285, 190, 116, 41, 13]#+[-1]*7

epi_curve = np.asarray(epi_curve,dtype=np.float64)
m_epi_curve = ma.masked_less_equal(epi_curve,0.,copy=True)


initial = [7,9,8,10,5,0,4,13]+[1]*(len(epi_curve)-8)
imported_cases = np.asarray(initial,dtype=np.float64)

startdate = datetime.date(2020,2,16)-datetime.timedelta(len(front))
mi = Modelinfo(startdate,epi_curve.shape[0])
ds = Datasets()
ds.AddDataSeries("epi_curve",m_epi_curve,startdate)
ds.AddDataSeries("initial",imported_cases,startdate)
print(mi)

trace = None
with pm.Model() as model:
     
    # Known working parameter Set: 1,.1 1,.1 t1=10,t2=24 offset=7 --> 2.53+/-0.050 / 0.99+/-0.070 
    # 6,24 o7 --> 0.97+/-0.021 / 0.83+/-0.066
 #   a1 = pm.Lognormal(name="initial_a1",mu=1,sigma=.1)
  #  a2 = pm.Lognormal(name="initial_a2",mu=1,sigma=.1)
   # imported_cases = GenInit(num_days_sim,a1,a2,t2=25,offset=5)
  #  pm.Deterministic("initial_t",imported_cases)
    
    initial_lambda = pm.Lognormal(
                name="lambda_0",
                mu=tt.log(pr_d["pr_median_lambda_0"]),
                sigma=pr_d["pr_sigma_lambda_0"],
           )
    # Weekend Factor
    f_weekend = 1.
    f_weekend = pm.Lognormal(name="f_weekend",mu=tt.log(f_weekend),sigma=.1)
    flt = np.array([.8,.3,.2,.1],dtype=np.float64)
    w_offset = mi.weekoffset
    lambda_t,week_mask = WeeklyRandomWalkWeekend("lambda_t",mi.length,initial_lambda,f_weekend,flt=flt,offset=w_offset)
    pm.Deterministic("lambda_t",lambda_t)
    
    median_incubation,sigma_incubation = 4.,.466
    # median_incubation = pm.Lognormal(
  #        name = "median_incubation",
  #        mu = tt.log(median_incubation),
  #        sigma = 0.04
  #   )
    
    # Core Function
    infected_t,infected_v,N_t = SEIR_model(86e6,imported_cases[:mi.length],lambda_t[:mi.length],median_incubation,sigma_incubation,l=32)
    onset_of_illness_t = est_deaths(infected_t,median=(4),sigma=.3,factor=1.,l=16)[:mi.length]
    pm.Deterministic("infected_t",infected_t)
    pm.Deterministic("onset_of_illness_t",onset_of_illness_t)
    
 #   f_wtrans = 0.15
    f_wtrans = pm.Uniform(name="w_trans",lower=0,upper=.5)
    reported_onset_of_illness_t = TransferWeekendReported(onset_of_illness_t,f_wtrans,week_mask)
    pm.Deterministic("reported_onset_of_illness_t",reported_onset_of_illness_t)
    
    
    sigma_onset_obs = pm.HalfCauchy( name="sigma_onset_obs", beta=10 )
    pm.StudentT(
            name="new_day_curve_studentT",
            nu=4,
            mu=reported_onset_of_illness_t,
            sigma=tt.abs_(reported_onset_of_illness_t + 1) ** 0.5 * sigma_onset_obs,  # +1 and tt.abs to avoid nans
            observed=m_epi_curve
       )
       
    
    delay_ratio_t = [.3]*20+[.4]*10+[.5]*10+[.6]*10+[.7]*20+[.8]*10+[.7]*20
    delay_m1,delay_s1,delay_m2,delay_s2 = 8,.4,20,.5
    reported_t,per_day_curves = reportDelayDistFunc(reported_onset_of_illness_t[:mi.length],
                                                                delay_m1,delay_s1,
                                                                delay_m2,delay_s2, delay_ratio_t[:mi.length],n=128)
    #    pm.Deterministic("delayed_distr",delay_distr)
    pm.Deterministic("reported_t",reported_t)
    
    #epi_curve_tc = epi_curve_t[4:num_days_sim+4]
    per_day_curves_t = per_day_curves[:mi.length,(mi.length-67):mi.length],
    per_day_curves_s = tt.cumsum(per_day_curves_t[0],axis=1 )
    pm.Deterministic("per_day_curve_t",per_day_curves)
    pm.Deterministic("per_day_curve_s",per_day_curves_s)
    
        
#init='advi'
#'advi+adapt_diag'
trace = pm.sample(model=model,init='advi+adapt_diag' , draws=200,cores=2,chains=2,tune=400)


if True:
    d = datetime.datetime.now()
    dn = "simple%02d%02d%02d"%(d.hour,d.minute,d.second)
    fn = "traces/trace"+dn+".dat"

    with open(fn,"wb+") as f:
        pickle.dump(trace,f)
        pickle.dump(mi,f)
        pickle.dump(ds,f)
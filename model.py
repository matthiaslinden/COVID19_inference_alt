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
#    dists = (d1.T*sr+(tt.ones_like(sr)-sr)*d2.T).T
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
# def est_deaths(infected_t,median,sigma,factor,l=40,offset=0):
#     beta = tt_lognormal(tt.arange(l*2),tt.log(median),sigma)
#     return conv_offset(infected_t,beta,factor,offset)

def WeeklyRandomWalk(name,n,initial,flt=np.array([.05,.1,.7,.1,.05],dtype=np.float64),sigma=.05,offset=0):
    offset = tt.cast(offset,"int64")
    delay_list_length = n//7+1
    rw_list = []
    rw_list.append(initial)
    sigma_random_walk = pm.HalfNormal(name=name+"_sigma_random_walk", sigma=sigma)
    delay_ratio_random_walk = pm.distributions.timeseries.GaussianRandomWalk(
                              name=name+"_random_walk",mu=0,
                              sigma=sigma_random_walk,shape=delay_list_length,
                              init=pm.Normal.dist(sigma=sigma),
                        )
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
pr_d["pr_median_lambda_0"] = 2.7
pr_d["pr_sigma_lambda_0"] = .5

# Data
#epi_curve = [0,0,0,0]+[7, 11, 8, 10, 34, 22, 26, 46, 65, 103, 147, 155, 195, 177, 193, 268, 255, 349, 363, 575, 744, 989, 1502, 1924, 2394, 2651, 3240, 3261, 3385, 4518, 3753, 3900, 3343, 3923, 3202, 2636, 3777, 2803, 3045, 2707, 2764, 2619, 2113, 2970, 2186, 2575, 2298, 2356, 1861, 1668, 2135, 1810, 1659, 1612, 1311, 1134, 1136, 1095, 1130, 1061, 933, 894, 795, 667, 834, 647, 614, 589, 515, 415, 351, 430, 295, 211, 160, 45, 18, 10, 2][:-1]
front = [0,0,0,0]
epi_curve = front+[7, 9, 8, 10, 33, 19, 26, 45, 64, 102, 147, 158, 197, 179, 195, 272, 258, 350, 366, 585, 747, 997, 1521, 1944, 2421, 2672, 3272, 3282, 3422, 4546, 3778, 3929, 3376, 3964, 3220, 2659, 3822, 2826, 3069, 2737, 2798, 2644, 2144, 2999, 2210, 2597, 2332, 2384, 1886, 1683, 2165, 1835, 1689, 1635, 1342, 1154, 1171, 1122, 1152, 1090, 959, 919, 819, 689, 874, 681, 641, 634, 564, 460, 402, 516, 370, 310, 274, 158, 125, 104, 69, 11, 1, 0, 0, 0]
epi_curve = np.asarray(epi_curve,dtype=np.float64)
epi_curve = ma.masked_less_equal(epi_curve,0.,copy=True)

startdate = datetime.date(2020,2,16)-datetime.timedelta(len(front))

initial_obs_rki_sit = [x*.6 for x in [9.188,9.188,4.594,4.594,22.971,13.783,27.565,27.565,27.565,53.507,32.821]]+[30.,16.,12.,20.,30.,40,40][:1]+[63.072,106.910,163.956,231.851,306.731,383.947,458.793,527.101,585.620,632.178,665.686,686.005,693.759,690.122,676.616,654.935,626.805,593.878,557.666,519.498,480.503,441.606,403.541,366.868,331.992,299.187,268.619,240.364,214.431,190.776,169.313,149.934,132.509,116.901,102.965,90.561,79.549,69.796,61.177,53.573,46.877,40.989,35.819,31.284,27.311,23.833,20.792,18.134,15.812,13.785,12.017,10.475,9.130,7.958,6.937,6.047,5.272,4.596,4.008,3.496,3.049,2.661,2.322,2.027,1.770]+[0]*7

initial_obs_rki_sit = [x*.5 for x in [9.188,9.188,4.594,4.594,22.971,13.783,27.565,27.565,27.565,23.507,12.821]]+[x*.7 for x in [30.,16.,12.,20.,30.,40]+[63.072,106.910,163.956,231.851,306.731,383.947,458.793,527.101,585.620,632.178,665.686,686.005,693.759,690.122,676.616,654.935,626.805,593.878,557.666,519.498,480.503,441.606,403.541,366.868,331.992,299.187,268.619,240.364,214.431,190.776,169.313,149.934,132.509,116.901,102.965,90.561,79.549,69.796,61.177,53.573,46.877,40.989,35.819,31.284,27.311,23.833,20.792,18.134,15.812,13.785,12.017,10.475,9.130,7.958,6.937,6.047,5.272,4.596,4.008,3.496,3.049,2.661,2.322,2.027,1.770]]+[0]*7

#initial_obs_sum_rki = [64,97,130,209,223,262]+[-1]*6+[3043]+[-1]*6+[9497]+[-1]*6+[13311]+[-1]*6+[14406]+[-1]*6+[14673]+[-1]*6+[14786]+[-1]*6+[14817]+[-1]*12
initial_obs_sum_rki = np.array([5, 14, 24, 31, 35, 36, 34, 32, 29, 28, 35, 55, 99, 172, 276, 402, 538, 664, 765, 828, 851, 834, 783, 710, 622, 530, 440, 357, 284, 222, 170, 129, 96, 71, 52, 37, 27, 19, 13, 9, 6, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],dtype=np.float64)
initial_obs_rki_sit = ma.masked_less(initial_obs_sum_rki,0.,copy=True)

#imported_cases = np.asarray(initial_obs_rki_sit[:len(epi_curve)],dtype=np.float64)
imported_cases = np.asarray([5,4,3,2,1]+[0]*80,dtype=np.float64)

hospital = [-1]*17+[-1]*20+[379,352,529,581,661,432,566,719,759,834,839,748,747,500,676,751,791,789,604,468,425,415,644,705,645,660,428,310,405,569,500,490,410,286,160,296,360,379,470,192,132,130]+[-1]*5
hospital_begin = 37
hospital = ma.masked_less(hospital,0.,copy=True)[hospital_begin:]

icu_deaths = [-1]*17+[-1]*36+[93,154,0,131,88,66,228,255,305,123,50,59,108,91,115,39,22,29,24,68,69,52,40,46,85]
icu_deaths_begin = 17+30
icu_deaths = ma.masked_less(icu_deaths,0.,copy=True)[icu_deaths_begin:]

deaths = [-1]*(17+4)+[-1,-1,-1,-1,-1,-1,0,1,2,0,3,4,0,0,0,8,11,16,8,31,28,35,49,55,72,64,66,128,149,140,145,141,184,92,173,254,246,266,171,129,126,170,285,315,299,242,184,110,194,281,215,227,179,140,110,163,202,173,193,94,74,43,139,165,123,147]+[-1]*5
deaths = np.asarray(deaths,dtype=np.float64)
deaths = ma.masked_less(deaths,0.,copy=True)

reported = [-1]*4+[66,138,239,156,107,237,157,271,802,693,733,400,1817,1144,1036,2807,2958,2705,1948,4062,4764,4118,4954,5780,6294,3965,4751,4615,5453,6156,6174,6082,5936,3677,3834,4003,4974,5323,4133,2821,2537,2082,2486,2866,3380,3609,2458,1775,1785,2237,2352,2337,2055,1737,1018,1144,1304,1478,1639,945,793,679,685,947]
reported = ma.masked_less(reported,0.,copy=True)


lambda_t = np.array( [2.045,2.050,2.050,2.054,2.056,2.059,2.064,2.067,2.071,2.076,2.080,2.083,2.082,2.080,2.074,2.071,2.068,2.065,2.060,2.052,2.043,2.033,2.019,1.996,1.953,1.895,1.823,1.731,1.628,1.527,1.406,1.285,1.163,1.070,0.968,0.897,0.867,0.806,0.780,0.755,0.752,0.746,0.752,0.783,0.772,0.793,0.790,0.787,0.754,0.740,0.759,0.748,0.731,0.714,0.685,0.663,0.662,0.667,0.682,0.682,0.680,0.679,0.669,0.665,0.677,0.670,0.672,0.677,0.680,0.683,0.694,0.718,0.750,0.774,0.789,0.793]+[.8]*4 ,dtype=np.float32)

print("Len of inputs",len(deaths),len(epi_curve),len(imported_cases),len(hospital),print(icu_deaths),len(reported))

#num_days_sim = min(len(epi_curve),len(imported_cases))
with open("data/onset_by_date_missing.pickle","rb") as f:
    onsets_per_date = pickle.load(f)
    print(onsets_per_date.shape)
    print(onsets_per_date[0])
    # date per line
    # days with report 

num_days_sim = onsets_per_date.shape[0]

print("num_days_sim",num_days_sim)

trace = None
with pm.Model() as model:
    # if False:
  #       lambda_list = []
  #       lambda_list.append(
  #           pm.Lognormal(
  #               name="lambda_0",
  #               mu=np.log(pr_d["pr_median_lambda_0"]),
  #               sigma=pr_d["pr_sigma_lambda_0"],
  #           )
  #       )
  #       # build the time-dependent spreading rate
  #       if pr_d.get("pr_sigma_random_walk",0) != 0:
  #           sigma_random_walk = pm.HalfNormal(
  #               name="sigma_random_walk", sigma=pr_d["pr_sigma_random_walk"]
  #           )
  #           lambda_t_random_walk = pm.distributions.timeseries.GaussianRandomWalk(
  #               name="lambda_t_random_walk",
  #               mu=0,
  #               sigma=sigma_random_walk,
  #               shape=num_days_sim,
  #               init=pm.Normal.dist(sigma=pr_d["pr_sigma_random_walk"]),
  #           )
  #           lambda_base = lambda_t_random_walk + lambda_list[0]
  #       else:
  #           lambda_base = lambda_list[0] * tt.ones(num_days_sim)
  #
  #       lambda_t_list = [lambda_base]
  #       # Transition-points
  #       # lambda_step_before = lambda_list[0]
  #       # for tr_begin, transient_len, lambda_step in zip(
  #       #     tr_begin_list, tr_len_list, lambda_list[1:]
  #       # ):
  #       #     lambda_t = mh.smooth_step_function(
  #       #         start_val=0,
  #       #         end_val=1,
  #       #         t_begin=tr_begin,
  #       #         t_end=tr_begin + transient_len,
  #       #         t_total=num_days_sim,
  #       #     ) * (lambda_step - lambda_step_before)
  #       #     lambda_step_before = lambda_step
  #       #     lambda_t_list.append(lambda_t)
  #
  #       lambda_t = sum(lambda_t_list)
     
    # Known working parameter Set: 1,.1 1,.1 t1=10,t2=24 offset=7 --> 2.53+/-0.050 / 0.99+/-0.070 
    # 6,24 o7 --> 0.97+/-0.021 / 0.83+/-0.066
 #   a1 = pm.Lognormal(name="initial_a1",mu=1,sigma=.1)
  #  a2 = pm.Lognormal(name="initial_a2",mu=1,sigma=.1)
   # imported_cases = GenInit(num_days_sim,a1,a2,t2=25,offset=5)
  #  pm.Deterministic("initial_t",imported_cases)
    
        
    initial_lambda = pm.Lognormal(
                name="lambda_0",
                mu=np.log(pr_d["pr_median_lambda_0"]),
                sigma=pr_d["pr_sigma_lambda_0"],
           )
#    flt = np.array([.2,.7,.7,1,.7,.7,.2],dtype=np.float64)
    # flt = np.array([1,.5,.2])
    flt = np.array([1],dtype=np.float64)
    f_weekend = 1.
    f_weekend = pm.Lognormal(name="f_weekend",mu=tt.log(f_weekend),sigma=.1)
    lambda_t,mask = WeeklyRandomWalkWeekend("lambda_t",num_days_sim,initial_lambda,f_weekend,flt=flt,offset=startdate.weekday())
    pm.Deterministic("lambda_t",lambda_t)
    
    initial_delay_ratio = pm.Uniform("delay_ratio_0",lower=.01,upper=.99)
#    delay_ratio_t = WeeklyRandomWalk("delay_ratio_t",num_days_sim,initial_delay_ratio,sigma=.01,flt=np.array([1,1,1,1,1,1,1],dtype=np.float64))
    delay_ratio_t = WeeklyRandomWalk("delay_ratio_t",num_days_sim,initial_delay_ratio,sigma=.01,flt=np.array([1,1,1,1,1,1,1],dtype=np.float64))
    
    delay_ratio_t = tt.clip(delay_ratio_t,.01,.99)
    pm.Deterministic("delay_ratio_t",delay_ratio_t)
    
    median_incubation,sigma_incubation = 4.,.466
    # median_incubation = pm.Lognormal(
  #        name = "median_incubation",
  #        mu = tt.log(median_incubation),
  #        sigma = 0.04
  #   )
    
    #
 #   s_import = 1.
 #   s_import = pm.Lognormal(name="median_imported_factor",mu=tt.log(1.),sigma=.05)
    
    # Core Function
    f_unknown = pm.Lognormal(name="f_unknown",mu=tt.log(1.1),sigma=.1)
    infected_t,infected_v,N_t = SEIR_model(86e6,imported_cases[:num_days_sim],lambda_t[:num_days_sim],median_incubation,sigma_incubation,l=32)
    pm.Deterministic("infected_t",infected_t)
    onset_of_illness_t = est_deaths(infected_t,median=(4),sigma=.25,factor=1.,l=12)[:num_days_sim]
    pm.Deterministic("onset_of_illness_t",onset_of_illness_t)
    
    delay_m1,delay_s1,delay_m2,delay_s2 = 10,.3,20,.4
    delay_m1 = pm.Lognormal(name="delay_m1",mu=tt.log(delay_m1),sigma=.2)
    delay_s1 = pm.HalfNormal(name="delay_s1",sigma=delay_s1)
   # delay_m2 = pm.Lognormal(name="delay_m2",mu=tt.log(delay_m2),sigma=.5)
    delay_s2 = pm.HalfNormal(name="delay_s2",sigma=delay_s2)

    reported_t,per_day_curves = reportDelayDistFunc(onset_of_illness_t[:num_days_sim],
                                                            delay_m1,delay_s1,
                                                            delay_m2,delay_s2, delay_ratio_t,n=128)
#    pm.Deterministic("delayed_distr",delay_distr)
    pm.Deterministic("reported_t",reported_t)
    
    #epi_curve_tc = epi_curve_t[4:num_days_sim+4]
    per_day_curves_t = per_day_curves[:num_days_sim,(num_days_sim-67):num_days_sim],
    per_day_curves_s = tt.cumsum(per_day_curves_t[0],axis=1)

    pm.Deterministic("per_day_curve_t",per_day_curves)
    pm.Deterministic("per_day_curve_s",per_day_curves_s)
    
    print(per_day_curves_s)
    print(onsets_per_date.shape)
    
    sigma_onset_obs = pm.HalfCauchy( name="sigma_onset_obs", beta=10 )
#    sigma_delayed_obs = pm.Lognormal(name="sigma_delayed_obs",mu=tt.log(10),sigma=.8)
    pm.StudentT(
            name="new_day_curve_studentT",
            nu=4,
            mu=per_day_curves_s,
            sigma=tt.abs_(per_day_curves_s + tt.alloc(1,per_day_curves_s.shape[1])) ** 0.5 * sigma_onset_obs,  # +1 and tt.abs to avoid nans
            observed=onsets_per_date
       )
    
    if False:
        sigma_reported_obs = pm.HalfCauchy( name="sigma_reorted_obs",beta=10 )
        pm.StudentT(
                name="new_reported_cases_studentT",
                nu=4,
                mu=onset_of_illness_t[14:],
                sigma=tt.abs_(onset_of_illness_t[14:] + 1) ** 0.5 * sigma_reported_obs,  # +1 and tt.abs to avoid nans
                observed=reported
            )
    
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
    if False:
        m_death,s_death,f_death = 25,.4,.04
        f_death = pm.Normal(name="f_death",mu=f_death,sigma=.01)
        m_death = pm.Normal(name="m_death",mu=m_death,sigma=1.)
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
#'advi+adapt_diag'
trace = pm.sample(model=model,init='advi+adapt_diag' , draws=50,cores=2,chains=2,tune=100)


if trace != None:
    d = datetime.datetime.now()
    ds = "%02d%02d%02d"%(d.hour,d.minute,d.second)
    fn = "traces/trace"+ds+".dat"

    with open(fn,"wb+") as f:
        pickle.dump(trace,f)
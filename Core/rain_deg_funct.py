#%% Rainflow

import rainflow as rf
import numpy as np
import pandas as pd

def MyFun(SOC_profile : np.array):

    # parameters 
    # non-linear degradation
    a_sei = 5.75e-2
    b_sei = 121
    #DoD stress
    k_d1 = 1.4e5
    k_d2 = -5.01e-1
    k_d3 = -1.23e5
    # SoC stress
    k_s = 1.04
    s_ref = 0.5 
    # temperature stress
    k_T = 6.93e-2
    T_ref = 25 #degC
    #calenar ageing
    k_t = 4.14e-10 # 1/second

    # functions
    funct_S_d = lambda d: (k_d1 * d ** k_d2 + k_d3)**(-1)  #DoD degradation
    funct_S_s = lambda s: np.exp(k_s*(s-s_ref))          #SOC degradation
    funct_S_T = lambda T: np.exp(k_T*(T-T_ref)*T_ref/T)  #Temperature degradation
    funct_S_t = lambda t: t*k_t                            #time degradation

    funct_f_cyc_i = lambda d, s, T: funct_S_d(d)* funct_S_s(s) * funct_S_T(T)   #cyclic ageing
    funct_f_cal = lambda s, t, T: funct_S_s(s) * funct_S_t(t) * funct_S_T(T)  #calendar ageing


    L = np.array([])
    L_sei = np.array([])


    rainflow = pd.DataFrame(columns=['Range', 'Mean', 'Count', 'Start', 'End'])

    for rng, mean, count, i_start, i_end in rf.extract_cycles(np.tile(SOC_profile, 1)): 
        new_row = pd.DataFrame({'Range': [rng], 'Mean': [mean], 'Count': [count], 'Start': [i_start], 'End': [i_end]})
        rainflow = pd.concat([rainflow, new_row], ignore_index=True)

    for i in range(1, 51):

        rnflow_data = rainflow.loc[rainflow.index.repeat((i-1)*8760/len(SOC_profile))]

        rf.count_cycles(SOC_profile)

        DoD = rnflow_data['Range']
        SOC = rnflow_data['Mean']
        f_cyc = funct_f_cyc_i(DoD, SOC, T_ref)*rnflow_data['Count'] #I multiply the weight of the cycle by the degradation of that cycle
        SOC_avg = SOC_profile.mean()
        f_cal = funct_f_cal(SOC_avg, 3600*SOC_profile.shape[0], T_ref)
        f_d = f_cyc.sum() + f_cal
        L = np.append(L, [1-np.exp(-f_d)])
        L_sei = np.append(L_sei, [1 - a_sei * np.exp(-b_sei*f_d) - (1-a_sei)*np.exp(-f_d)])
    
    return np.argmax(L_sei >= 0.2) + 1, 1-L_sei #the first is the cyclelife of the battery for a given SOC profile of 1 year, supposing that the battery periodically implements that strategy
                                            #the secon one is the SOH of the battery every year 


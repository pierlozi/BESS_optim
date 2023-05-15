import numpy as np
import matplotlib.pyplot as plt

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

SOC_try = 0.5
DoD_try = 0.7
T_try = 25

t_cycle = 24 #hours

f_d_1cyc = funct_f_cal(SOC_try, t_cycle, T_try) + funct_f_cyc_i(DoD_try, SOC_try, T_try)

t_op = 8760
N = 4000
N_cyc = np.linspace(0,int(N),int(N)+1)
L = 1-np.exp(-N_cyc*f_d_1cyc)
L_sei = 1 - a_sei * np.exp(-b_sei*N_cyc*f_d_1cyc) - (1-a_sei)*np.exp(-N_cyc*f_d_1cyc)
plt.plot(N_cyc, 1-L)
plt.plot(N_cyc, 1-L_sei)

plt.legend(['L', 'L_sei'])


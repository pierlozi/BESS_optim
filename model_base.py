
# %%

import time
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
size_font = 10

optim_time = 8760 # number of hours to display
time_range_optim = range(optim_time)

''' Reading the data from csv files'''
price_data = pd.read_csv("PriceCurve_SE3_2021.csv", header = 0, nrows = 8760, sep=';') #eur/MWh
P_load_data = pd.read_excel('load_data.xlsx', sheet_name='Yearly Load', header=0) #MW
P_ren_read = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760) #W

surplus_perc = []

'''Initializing the lists of all the paramters and variables which values I want to store'''

SOC = []
P_BES = []
P_dch = []
P_dg = []
P_curt = []

BES_cost = []
Er_BES =  []
Pr_BES = []
   
dg_cost = []    
Pr_dg = []

'''I multiply the renewable power production with a scaling factor to bring it to the same order of magnitutde of the load'''
P_prod_data = P_ren_read['Power']/1e6*7 #MW 

'''I produce dictionaries of the data imported as the pyomo framework reads input parameters in form of dictionary'''
price_dict = dict()
for i in time_range_optim:
    price_dict[i] = price_data['Grid_Price'].values[i]

P_load_dict = dict() 
for i in time_range_optim:
    P_load_dict[i] = P_load_data['Load [MW]'].values[i]
    
P_prod_dict = dict()
for i in time_range_optim:
    P_prod_dict[i] = P_prod_data[i] #MW

    
# ax_pow = plt.gca()

# P_load.iloc[0:len(price)].plot(kind="line", y = 'Load [MW]', ax = ax_pow)
# P_ren.iloc[0:len(price)].plot(kind="line", y = 'Power', ax = ax_pow)

# plt.show()

# ren_surplus = sum(P_ren_data['Power'].values[i] for i in time_range_optim)-sum(P_load_data['Load [MW]'].values[i] for i in time_range_optim)
# print('The renewable energy surplus is of ',ren_surplus , 'MWh')

''' bess_bin is the variable that tells which type of microgrid we have, 0 = only diesel ; 1 = diesel + res; 2 = diesel + res + storage.
I use it so that the model compiles different objective functions and constraints for the two types.'''

# %%

m = ConcreteModel()

#m.iIDX is the set which keeps the time in the simulation
m.iIDX = Set(initialize = time_range_optim)

'''importing data in the pyomo framewrok''' 
m.P_load = Param(m.iIDX,initialize=P_load_dict)
m.P_prod = Param(m.iIDX, initialize=P_prod_dict)
m.price = Param(m.iIDX, initialize = price_dict)

'''initializing parameters for the simulation'''

#the charging and dischargin efficiencies are calculeted using a roundrtip efficiency value
#of 0.95 (Reference: Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year)
m.eff_ch = Param(initialize=sqrt(0.95)) 
m.eff_dch = Param(initialize=sqrt(0.95))

# m.lifetime of the battery is the float life of the battery (Reference: Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year)
m.lifetime = Param(initialize=10) #years
m.C_P = Param(initialize=320) #$/kW
m.C_E = Param(initialize=360) #$/kWh
m.C_inst = Param(initialize=15) #$/kWh
m.C_OM = Param(initialize=5) #$/kW
m.sigma = Param(initialize=0.002/24) #original daily self discharge is 0,2% -> we need an hourly self discharge

m.gamma_min = Param(initialize=0)
m.gamma_MAX = Param(initialize=10000) #maximum 10 (Stefan) -> for now I leave it very high to see how the system behaves

m.P_BES_MAX = Param(initialize=5*max(P_load_data['Load [MW]']))

m.price_f = Param(initialize=2) #euro/L

#empirical parameters for diesel fuel consumption from 
# "Multi objective particle swarm optimization of hybrid micro-grid system: A case study in Sweden"
m.alpha = Param(initialize=0.24) #L/kW
m.beta = Param(initialize=0.084) #L/kW

#minimum up and down time for diesel from
# "Optimal sizing of battery energy storage systems in off-grid micro grids using convex optimization"
m.UT = Param(initialize = 5) #h
m.DT = Param(initialize = 20) #h

m.Pr_dg_MAX = Param(initialize = max(P_load_data['Load [MW]']))

#with the BESS    
m.P_ch = Var(m.iIDX, domain=NonNegativeReals)
m.P_dch = Var(m.iIDX, domain = NonNegativeReals)
m.P_RES = Var(m.iIDX, domain = NonNegativeReals)
m.P_curt = Var(m.iIDX, domain = NonNegativeReals)
m.Pr_BES = Var(domain=NonNegativeReals, bounds=(0, 10*max(P_load_data['Load [MW]'])))
m.Er_BES = Var(domain=NonNegativeReals)
m.SOC = Var(m.iIDX, domain=NonNegativeReals)
m.SOC_ini = Var(domain=NonNegativeReals)

m.bin_dch = Var(m.iIDX, domain=Binary)


m.P_dg = Var(m.iIDX, domain = NonNegativeReals) #hourly power of diesel
m.Pr_dg = Var(domain=NonNegativeReals) #power rating of diesel
m.v_dg = Var(m.iIDX, domain = Binary) #1 when dg turned on at timestep
m.w_dg = Var(m.iIDX, domain = Binary) #1 when dg turned off at timestep
m.u_dg = Var(m.iIDX, domain=Binary) # commitment of unit (1 if unit is on)

 # %%    
def obj_funct(m):
    return (m.Pr_BES*m.C_P + m.Er_BES*(m.C_E+m.C_inst))*1e3 + (m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX))*m.lifetime

m.obj = Objective(rule = obj_funct, sense=minimize)

def f_equi_RES(m,i):
    return m.P_prod[i] == m.P_RES[i] + m.P_ch[i] + m.P_curt[i]

m.cstr_eq_RES = Constraint(m.iIDX, rule = f_equi_RES)

def f_equi_load(m,i):
    return m.P_dch[i] + m.P_RES[i] + m.P_dg[i] == m.P_load[i]

m.cstr_eq_load = Constraint(m.iIDX, rule = f_equi_load)
 

'BATTERY CONSTRAINTS'#---------------------------------------------------------------------------------------------------------------

def f_SOC_lim_up(m,i):
    return m.SOC[i]<= m.Er_BES

m.cstr_SOC_lim_up = Constraint(m.iIDX, rule=f_SOC_lim_up)

# def f_SOC_lim_low(m,i):
#     return m.SOC[i]>= 0.2*m.Er_BES

# m.cstr_SOC_lim_low = Constraint(m.iIDX, rule=f_SOC_lim_low)

def f_SOC_ini_lim(m):
    return m.SOC_ini <= m.Er_BES

m.cstr_SOC_ini_lim = Constraint(rule=f_SOC_ini_lim)

def f_SOC(m,i):
    if i == 0:
        return m.SOC[i] == m.SOC_ini
    else:
        return m.SOC[i] == m.SOC[i-1]*(1-m.sigma) + m.P_ch[i]*m.eff_ch - m.P_dch[i]/m.eff_dch 

m.cstr_SOC = Constraint(m.iIDX, rule = f_SOC)

m.cstr_SOC_final = Constraint(expr = m.SOC[len(m.iIDX)-1]==m.SOC_ini )

def f_dch_lim(m,i):
    return m.P_dch[i] <= m.Pr_BES

m.cstr_dch_lim = Constraint(m.iIDX, rule = f_dch_lim)

def f_dch_bin(m,i):
    return m.P_dch[i] <= m.bin_dch[i]*m.P_BES_MAX

m.cstr_dch_bin = Constraint(m.iIDX, rule=f_dch_bin)

def f_ch_lim(m,i):
    return m.P_ch[i] <= m.Pr_BES

m.cstr_ch_lim = Constraint(m.iIDX, rule = f_ch_lim)

def f_ch_bin(m,i):
    return m.P_ch[i] <= (1-m.bin_dch[i])*m.P_BES_MAX

m.cstr_ch_bin = Constraint(m.iIDX, rule=f_ch_bin)

def f_Er_BES_lim(m):
    return m.Er_BES <= m.gamma_MAX*m.Pr_BES

m.cstr_Er_BES_lim = Constraint(rule=f_Er_BES_lim)

'----------------------------------------------------------------------------------------------------------------------------'
# %% 
def f_dg_lim(m,i):
    return m.P_dg[i] <= m.Pr_dg

m.cstr_dg_lim = Constraint(m.iIDX, rule=f_dg_lim)

def f_dg_commit(m, i):
    return m.P_dg[i] <= m.u_dg[i]*m.Pr_dg_MAX

m.cstr_dg_commit = Constraint(m.iIDX, rule=f_dg_commit)


m.cstr_dg_uptime = ConstraintList()

# def f_dg_uptime(m, i):
#     return sum(m.u[j] for j in range(i, i + m.UT) ) >= m.UT*m.v_dg[i]

for i in m.iIDX:
    if i <= len(m.iIDX) - m.UT - 1:
        m.cstr_dg_uptime.add(sum(m.u_dg[j] for j in range(i, i + m.UT) ) >= m.UT*m.v_dg[i])

m.cstr_dg_dwntime = ConstraintList()

# def f_dg_dwntime(m, i):
#     return sum((1 - m.u_dg[j]) for j in range(i, i + m.UT) ) >= m.DT*m.w_dg[i]

for i in m.iIDX:
    if i <= len(m.iIDX) - m.DT - 1:
        m.cstr_dg_dwntime.add(sum((1 - m.u_dg[j]) for j in range(i, i + m.DT) ) >= m.DT*m.w_dg[i]
)   
m.cstr_up_dwn_commit = ConstraintList()

# def f_up_dwn_commit(m, i):
#     return m.v_dg[i] - m.w_dg[i] == m.u_dg[i] - m.u_dg[i-1]

for i in m.iIDX:
    if i > 0:
        m.cstr_up_dwn_commit.add(m.v_dg[i] - m.w_dg[i] == m.u_dg[i] - m.u_dg[i-1])

def f_up_dwn_excl(m, i):
    return m.v_dg[i] + m.w_dg[i] <= 1

m.cstr_up_dwn_excl = Constraint(m.iIDX, rule = f_up_dwn_excl)

# %%
start = time.time()
opt = SolverFactory("gurobi")
opt.solve(m)
code_time = time.time() - start

# %%

P_prod = np.array([value(m.P_prod[i]) for i in m.iIDX])
P_load = np.array([value(m.P_load[i]) for i in m.iIDX])
     
SOC.append(np.array([value(m.SOC[i]) for i in m.iIDX])/value(m.Er_BES))
P_BES.append(np.array([(value(m.P_dch[i]) - value(m.P_ch[i]) ) for i in m.iIDX]))
P_dch.append(value(sum(m.P_dch[i] for i in m.iIDX)))
BES_cost.append(value((m.Pr_BES*m.C_P + m.Er_BES*(m.C_E+m.C_inst))*1e3)/1e6) #milions
Er_BES.append(value(m.Er_BES))
Pr_BES.append(value(m.Pr_BES))
    
P_curt.append(np.array([value(m.P_curt[i]) for i in m.iIDX]))
    

P_dg.append(np.array([value(m.P_dg[i]) for i in m.iIDX]))  
dg_cost.append(value((m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX))*10/1e6))    
Pr_dg.append(value(m.Pr_dg))
u_dg = np.array([value(m.u_dg[i]) for i in m.iIDX])

# LCOS = (BES_cost[n_month-1] + sum((Er_BES[n_month-1]*value(m.C_OM)*1e3/(1+value(m.IR))**(y-1)) 
#                                       for y in range(1,value(m.lifetime)+1)))/ \
#             sum((P_dch/(1+value(m.IR))**(y-1)) for y in range(1,value(m.lifetime)+1))

# print('The code took ', code_time ,' s.')
# print('The optimal storage capacity is ', Er_BES,' MWh, with a rated power of ', Pr_BES,'MW.')
# print('The optimal dg power rating is', Pr_dg,' MW.')
# print('The total cost of the battery system is ', BES_cost, 'millions.')
# print('The total cost related to diesel fuel expense is ', dg_cost, 'millions.')


P_BES = np.asarray(P_BES)
P_curt = np.asarray(P_curt)
P_dg = np.asarray(P_dg)
SOC = np.asarray(SOC)

# %%

# Here I build a dataframe to better visually show the results
data = pd.DataFrame({'Er_BES [MWh]': Er_BES,
                     'Pr_BES [MW]': Pr_BES,
                     'Pr_diesel [MW]': Pr_dg,
                     'BES cost [million euros]': BES_cost,
                     'Fuel cost [million euros]': dg_cost,
                     })

# data = data.reset_index(drop=True)
data['Total cost [million euros]'] = data['BES cost [million euros]'] + data['Fuel cost [million euros]']

print(data.T)

# %% DATA PLOT

display_start = 500
display_end = 600

time_horizon = range(display_start, display_end)
    
# fig, ax_pow = plt.subplots(figsize=(10,8))

# ax_pow.set_ylabel("Power [MW]",fontsize=size_font)
# ax_pow.set_xlabel("time_horizon [h]",fontsize=size_font)
# ax_pow.plot(time_horizon, P_load[display_start:display_end], color='black')
# ax_pow.plot(time_horizon, P_BES[display_start:display_end], color = 'blue')
# ax_pow.plot(time_horizon, P_ren[display_start:display_end], color = 'green')
# ax_pow.plot(time_horizon, P_curt[display_start:display_end], color='red')
# ax_pow.legend(['P_Load', 'P_BES', 'P_ren', 'P_curt'],loc=4)


# ax_pow.spines['right'].set_color("black")


# plt.grid(True)

# fig.set_size_inches(10,8)
# fig.set_dpi(200)
# ax_pow.tick_params(axis='both', which='major', labelsize=size_font)

# # plt.savefig('powers_price.png',bbox_inches='tight', dpi=150)

# plt.show()
# plt.close()

fig, ax = plt.subplots()

ax.plot(time_horizon, u_dg[display_start:display_end])
ax.plot(time_horizon, P_dg[0,display_start:display_end], color='orange')

plt.show()

fig, ax_pow = plt.subplots(figsize=(10,8))

ax_pow.set_ylabel("Power [MW]",fontsize=size_font)
ax_pow.set_xlabel("Time [h]",fontsize=size_font)
ax_pow.plot(time_horizon, P_load[display_start:display_end], color='black')
ax_pow.plot(time_horizon, P_BES[0,display_start:display_end], color = 'blue')
ax_pow.plot(time_horizon, P_prod[display_start:display_end], color = 'green')
ax_pow.plot(time_horizon, P_curt[0,display_start:display_end], color='red')
ax_pow.plot(time_horizon, P_dg[0,display_start:display_end], color='orange')
ax_pow.legend(['P_Load', 'P_BES', 'P_RES', 'P_curt','P_diesel'],loc=4)


ax_pow.set_title("Energy dispatch diesel + PV + BESS")

ax_SOC = ax_pow.twinx()  # instantiate a second axes that shares the same x-axis

ax_SOC.set_ylabel('SOC [-]',fontsize=size_font)  # we already handled the x-label with ax1
ax_SOC.plot(time_horizon, SOC[0,display_start:display_end], color='cyan')
ax_SOC.legend(['SOC'],loc=2)

ax_pow.spines['right'].set_position(('axes',0.15))

ax_pow.spines['right'].set_color("black")
ax_SOC.spines['right'].set_color("cyan")

plt.grid(True)


fig.set_size_inches(10,8)
fig.set_dpi(200)
ax_pow.tick_params(axis='both', which='major', labelsize=size_font)
ax_SOC.tick_params(axis='both', which='major', labelsize=size_font)

# plt.savefig('powers_SOC.png',bbox_inches='tight', dpi=150)

plt.show()

plt.close()

# fig, ax_pow = plt.subplots(figsize=(10,8))

# ax_pow.set_ylabel("Energy throughput [% of nominal capacity]",fontsize=size_font)
# ax_pow.set_xlabel("Time [h]",fontsize=size_font)
# ax_pow.set_title("Energy throughput")
# ax_pow.plot(time_horizon, abs(P_BES[display_start:display_end])/Er_BES[-1]*100, color = 'blue')




# plt.grid(True)


# fig.set_size_inches(10,8)
# fig.set_dpi(200)
# ax_pow.tick_params(axis='both', which='major', labelsize=size_font)
# # ax_SOC.tick_params(axis='both', which='major', labelsize=size_font)

# # plt.savefig('powers_SOC.png',bbox_inches='tight', dpi=150)

# plt.show()
# plt.close()



# %%

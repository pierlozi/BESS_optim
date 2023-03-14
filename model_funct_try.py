#%%
import time

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator
import matplotlib.dates as mdates

from math import sqrt

from matplotlib.pyplot import figure


class microgrid():

    def __init__(self, Er_BES, Pr_BES, P_load, P_ren):
        self.Er_BES = Er_BES
        self.Pr_BES = Pr_BES
        self.P_load = P_load
        self.P_ren = P_ren

def dispatcher_GA(design, bin_var): #bin_var is to tell if power and energy rating are variables (bin_var=1) or input parameters (bin_var=0) of the dispatcher
        
    mine_life = 13

    optim_time = 8760 # number of hours to display
    time_range_optim = range(optim_time)

    
    '''Initializing the lists of all the paramters and variables which values I want to store'''
    code_time = []

    '''Initializing the lists of all the paramters and variables whose values I want to store'''

    Er_BES = []
    Pr_BES = []

    SOC = []
    P_BES = []
    P_dch = []
    P_dg = []
    P_curt = []

    BES_capex = [] #installation and replacement cost
    BES_opex = [] #operation and maintenance cost

    P_thr = []

    dg_opex = []    
    Pr_dg = []

    BES_cyclelife = []
    cycles_y = []
    dsctd_cash_flows = []

    '''I multiply the renewable power production with a scaling factor to bring it to the same order of magnitutde of the load'''
    P_prod_data = design.P_ren['Power']*7/1e6 #MW 

    '''I produce dictionaries of the data imported as the pyomo framework reads input parameters in form of dictionary'''
    P_load_dict = dict() 
    for i in time_range_optim:
        P_load_dict[i] = design.P_load['Load [MW]'].values[i]
        
    P_prod_dict = dict()
    for i in time_range_optim:
        P_prod_dict[i] = P_prod_data[i] #MW


    '''Here I set the max power of the BESS'''
    P_BES_MAX = 10*max(max(P_load_data['Load [MW]']),max(P_prod_data))

    ''' bess_bin is the variable that tells which type of microgrid we have, 0 = only diesel ; 1 = diesel + res; 2 = diesel + res + storage.
    I use it so that the model compiles different objective functions and constraints for the two types.'''


    m = pyo.ConcreteModel()

    #m.iIDX is the set which keeps the time in the simulation
    m.iIDX = pyo.Set(initialize = time_range_optim)

    '''importing data in the pyomo framewrok''' 
    m.P_load = pyo.Param(m.iIDX,initialize=P_load_dict)
    m.P_prod = pyo.Param(m.iIDX, initialize=P_prod_dict)
    #m.price = pyo.Param(m.iIDX, initialize = price_dict)

    '''initializing parameters for the simulation'''

    #the charging and dischargin efficiencies are calculeted using a roundrtip efficiency value
    #of 0.95 (Reference: Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year)
    m.eff_ch = pyo.Param(initialize=sqrt(0.95)) 
    m.eff_dch = pyo.Param(initialize=sqrt(0.95))

    # data from 'Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year'

    # m.floatlife = pyo.Param(initialize=10) #years
    # m.C_P = pyo.Param(initialize=320) #$/kW
    # m.C_E = pyo.Param(initialize=360) #$/kWh
    # m.C_inst = pyo.Param(initialize=15) #$/kWh
    # m.C_POM = pyo.Param(initialize=5) #$/kW operation cost related to power
    # m.C_EOM = pyo.Param(initialize=0) #$/Mwh operation cost related to energy
    # m.sigma = pyo.Param(initialize=0.002/24) #original daily self discharge is 0,2% -> we need an hourly self discharge
    # m.IR = pyo.Param(initialize = 5/100)

    #data from 'Projecting the Future Levelized Cost of Electricity Storage Technologies'
    m.floatlife = pyo.Param(initialize=10) #years
    m.C_P = pyo.Param(initialize=678) #$/kW
    m.C_E = pyo.Param(initialize=802) #$/kWh
    m.C_inst = pyo.Param(initialize=0) #$/kWh (the reference doesnt take into account installation)
    m.C_POM = pyo.Param(initialize=10) #$/kW operation cost related to power
    m.C_EOM = pyo.Param(initialize=3) #$/Mwh operation cost related to energy
    m.sigma = pyo.Param(initialize=0) #original daily self discharge is 0,2% -> we need an hourly self discharge
    m.IR = pyo.Param(initialize = 8/100)

    # '''Adding the table of DoD - cycle life to implement battery degradation'''
    m.DoD = pyo.Param(initialize = 75)
    m.cyclelife = pyo.Param(initialize = 2700)

    m.gamma_min = pyo.Param(initialize=0)
    m.gamma_MAX = pyo.Param(initialize=100000) #maximum 10 (Stefan) -> for now I leave it very high to see how the system behaves

    m.P_BES_MAX = pyo.Param(initialize=P_BES_MAX)

    m.price_f = pyo.Param(initialize=1.66) #euro/L

    #empirical parameters for diesel fuel consumption from 
    # "Multi objective particle swarm optimization of hybrid micro-grid system: A case study in Sweden"
    m.alpha = pyo.Param(initialize=0.24) #L/kW
    m.beta = pyo.Param(initialize=0.084) #L/kW

    #minimum up and down time for diesel from
    # "Optimal sizing of battery energy storage systems in off-grid micro grids using convex optimization"
    m.UT = pyo.Param(initialize = 5) #h
    m.DT = pyo.Param(initialize = 1) #h

    #max and min power rating of the diesel generator, I choose it arbitrarily
    m.Pr_dg_MAX = pyo.Param(initialize = max(P_load_data['Load [MW]']))
    m.Pr_dg_MIN = pyo.Param(initialize = 0.1*sum(P_load_data['Load [MW]'])/len(P_load_data['Load [MW]']))



    #with the BESS    
    m.P_ch = pyo.Var(m.iIDX, domain=pyo.NonNegativeReals)
    m.P_dch = pyo.Var(m.iIDX, domain = pyo.NonNegativeReals)
    m.P_RES = pyo.Var(m.iIDX, domain = pyo.NonNegativeReals)
    m.P_curt = pyo.Var(m.iIDX, domain = pyo.NonNegativeReals)
    
    if bin_var:
        m.Pr_BES = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, P_BES_MAX))
        m.Er_BES = pyo.Var(domain=pyo.NonNegativeReals)
    else:
        m.Pr_BES = pyo.Param(initialize = design.Pr_BES)
        m.Er_BES = pyo.Param(initialize = design.Er_BES)

    
    m.SOC = pyo.Var(m.iIDX, domain=pyo.NonNegativeReals)
    m.SOC_ini = pyo.Var(domain=pyo.NonNegativeReals)

    # '''this is the bi-linear variable used to implement the DoD-cyclelife constraint'''
    # m.LEr = pyo.Var(m.bIDX, domain = NonNegativeReals)
    # m.chi = pyo.Var(m.bIDX, domain = Binary)

    m.bin_dch = pyo.Var(m.iIDX, domain=pyo.Binary)


    m.P_dg = pyo.Var(m.iIDX, domain = pyo.NonNegativeReals) #hourly power of diesel
    m.Pr_dg = pyo.Var(domain=pyo.NonNegativeReals) #power rating of diesel

    # # these are the binary variables to be used for the min up/down times of the diesel generator

    # m.v_dg = pyo.Var(m.iIDX, domain = Binary) #1 when dg turned on at timestep
    # m.w_dg = pyo.Var(m.iIDX, domain = Binary) #1 when dg turned off at timestep
    # m.u_dg = pyo.Var(m.iIDX, domain=Binary) # commitment of unit (1 if unit is on)



    #  OBJ and Microgrid  
    if bin_var:
        def obj_funct(m): #7 is the expected years lifetime of the BES with 75% DoD
            return (m.Pr_BES*(m.C_P + 10*m.C_POM) + m.Er_BES*(m.C_E+m.C_inst+10*m.C_EOM/1e3))*1e3 + (m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX))*7 
    else:
        def obj_funct(m):
            return m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX)*7

    m.obj = pyo.Objective(rule = obj_funct, sense = pyo.minimize)

    def f_equi_RES(m,i):
        return m.P_prod[i] == m.P_RES[i] + m.P_ch[i] + m.P_curt[i]

    m.cstr_eq_RES = pyo.Constraint(m.iIDX, rule = f_equi_RES)

    def f_equi_load(m,i):
        return m.P_dch[i] + m.P_RES[i] + m.P_dg[i] == m.P_load[i]

    m.cstr_eq_load = pyo.Constraint(m.iIDX, rule = f_equi_load)
    

    # 'BATTERY CONSTRAINTS'#---------------------------------------------------------------------------------------------------------------

    def f_SOC_lim_up(m,i):
        return m.SOC[i]<= m.Er_BES

    m.cstr_SOC_lim_up = pyo.Constraint(m.iIDX, rule=f_SOC_lim_up)

    def f_SOC_lim_low(m,i):
        return m.SOC[i]>= m.Er_BES * ( 1 - m.DoD/100)

    m.cstr_SOC_lim_low = pyo.Constraint(m.iIDX, rule=f_SOC_lim_low)

    def f_SOC_ini_lim(m):
        return m.SOC_ini <= m.Er_BES

    m.cstr_SOC_ini_lim = pyo.Constraint(rule=f_SOC_ini_lim)

    def f_SOC(m,i):
        if i == 0:
            return m.SOC[i] == m.SOC_ini
        else:
            return m.SOC[i] == m.SOC[i-1]*(1-m.sigma) + m.P_ch[i]*m.eff_ch - m.P_dch[i]/m.eff_dch 

    m.cstr_SOC = pyo.Constraint(m.iIDX, rule = f_SOC)

    m.cstr_SOC_final = pyo.Constraint(expr = m.SOC[len(m.iIDX)-1]==m.SOC_ini )

    def f_dch_lim(m,i):
        return m.P_dch[i] <= m.Pr_BES

    m.cstr_dch_lim = pyo.Constraint(m.iIDX, rule = f_dch_lim)

    def f_dch_bin(m,i):
        return m.P_dch[i] <= m.bin_dch[i]*m.P_BES_MAX

    m.cstr_dch_bin = pyo.Constraint(m.iIDX, rule=f_dch_bin)

    def f_ch_lim(m,i):
        return m.P_ch[i] <= m.Pr_BES

    m.cstr_ch_lim = pyo.Constraint(m.iIDX, rule = f_ch_lim)

    def f_ch_bin(m,i):
        return m.P_ch[i] <= (1-m.bin_dch[i])*m.P_BES_MAX

    m.cstr_ch_bin = pyo.Constraint(m.iIDX, rule=f_ch_bin)


    '----------------------------------------------------------------------------------------------------------------------------'
    # DG constraints
    def f_dg_lim(m,i):
        return m.P_dg[i] <= m.Pr_dg

    m.cstr_dg_lim = pyo.Constraint(m.iIDX, rule=f_dg_lim)

    # def f_dg_commit_sup(m, i):
    #     return m.P_dg[i] <= m.u_dg[i]*m.Pr_dg_MAX

    # m.cstr_dg_commit_sup = pyo.Constraint(m.iIDX, rule=f_dg_commit_sup)

    # def f_dg_commit_inf(m, i):
    #     return m.P_dg[i] >= m.u_dg[i]*m.Pr_dg_MIN

    # m.cstr_dg_commit_inf = pyo.Constraint(m.iIDX, rule=f_dg_commit_inf)


    # m.cstr_dg_uptime = pyo.ConstraintList()

    # # def f_dg_uptime(m, i):
    # #     return sum(m.u[j] for j in range(i, i + m.UT) ) >= m.UT*m.v_dg[i]

    # for i in m.iIDX:
    #     if i <= len(m.iIDX) - m.UT - 1:
    #         m.cstr_dg_uptime.add(sum(m.u_dg[j] for j in range(i, i + m.UT) ) >= m.UT*m.v_dg[i])

    # m.cstr_dg_dwntime = pyo.ConstraintList()

    # # def f_dg_dwntime(m, i):
    # #     return sum((1 - m.u_dg[j]) for j in range(i, i + m.UT) ) >= m.DT*m.w_dg[i]

    # for i in m.iIDX:
    #     if i <= len(m.iIDX) - m.DT - 1:
    #         m.cstr_dg_dwntime.add(sum((1 - m.u_dg[j]) for j in range(i, i + m.DT) ) >= m.DT*m.w_dg[i]
    # )   
    # m.cstr_up_dwn_commit = pyo.ConstraintList()

    # # def f_up_dwn_commit(m, i):
    # #     return m.v_dg[i] - m.w_dg[i] == m.u_dg[i] - m.u_dg[i-1]

    # for i in m.iIDX:
    #     if i > 0:
    #         m.cstr_up_dwn_commit.add(m.v_dg[i] - m.w_dg[i] == m.u_dg[i] - m.u_dg[i-1])

    # def f_up_dwn_excl(m, i):
    #     return m.v_dg[i] + m.w_dg[i] <= 1

    # m.cstr_up_dwn_excl = pyo.Constraint(m.iIDX, rule = f_up_dwn_excl)

    # Initializing results lists

    start = time.time()
    opt = pyo.SolverFactory("gurobi")
    opt.solve(m)
    code_time.append( time.time() - start)


    P_prod = np.array([pyo.value(m.P_prod[i]) for i in m.iIDX])
    P_load = np.array([pyo.value(m.P_load[i]) for i in m.iIDX])
        
    SOC.append(np.array([pyo.value(m.SOC[i]) for i in m.iIDX])/pyo.value(m.Er_BES))
    P_BES.append(np.array([(pyo.value(m.P_dch[i]) - pyo.value(m.P_ch[i]) ) for i in m.iIDX]))
    P_dch.append(pyo.value(sum(m.P_dch[i] for i in m.iIDX)))
    BES_capex.append(pyo.value((m.Pr_BES*m.C_P + m.Er_BES*(m.C_E+m.C_inst))*1e3)) #€
    BES_opex.append(pyo.value(m.Pr_BES*m.C_POM*1e3 + m.Er_BES*m.C_EOM)) #€/year
    Er_BES.append(pyo.value(m.Er_BES))
    Pr_BES.append(pyo.value(m.Pr_BES))

    # chi = np.array([value(m.chi[b]) for b in m.bIDX])
    # LEr = np.array([value(m.LEr[b]) for b in m.bIDX])
        
    P_curt.append(np.array([pyo.value(m.P_curt[i]) for i in m.iIDX]))

    P_dg.append(np.array([pyo.value(m.P_dg[i]) for i in m.iIDX]))  
    dg_opex.append(pyo.value((m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX)))) #€/year   
    Pr_dg.append(pyo.value(m.Pr_dg))
    # u_dg = np.array([pyo.value(m.u_dg[i]) for i in m.iIDX])
    # w_dg = np.array([pyo.value(m.w_dg[i]) for i in m.iIDX])
    # v_dg = np.array([pyo.value(m.v_dg[i]) for i in m.iIDX])

    P_thr.append(abs(P_BES[-1]))

    if pyo.value(m.Er_BES) != 0 and pyo.value(m.Pr_BES) != 0:
        cycles_y.append(sum(P_thr[-1])/Er_BES[-1])

        BES_cyclelife.append(round(pyo.value(m.cyclelife)/cycles_y[-1]))

        cost_cash_flow  = []
        for i in range(0, mine_life):
            if mine_life > pyo.value(m.floatlife): #if the mine has a life longer than shelflife of battery
                if BES_cyclelife[-1] >= pyo.value(m.floatlife): #if battery has to be changed at floatlife
                    if i == 0:
                        cost_cash_flow.append(BES_capex[-1] + BES_opex[-1]) # + dg_opex[-1]) #€
                    elif i == pyo.value(m.floatlife) - 1:    
                        cost_cash_flow.append(BES_capex[-1] + BES_opex[-1]) # + dg_opex[-1]) #€
                    else:
                        cost_cash_flow.append(BES_opex[-1]) # + dg_opex[-1]) #€
                else: #if the battery has to be changed at cycle life
                    if i == 0:
                        cost_cash_flow.append(BES_capex[-1] + BES_opex[-1]) # + dg_opex[-1]) #€
                    elif i == BES_cyclelife[-1] - 1:    
                        cost_cash_flow.append(BES_capex[-1] + BES_opex[-1]) # + dg_opex[-1]) #€
                    else:
                        cost_cash_flow.append(BES_opex[-1]) # + dg_opex[-1]) #€
            else: #if the battery has a longer life than the mine
                if i == 0:
                    cost_cash_flow.append(BES_capex[-1] + BES_opex[-1]) # + dg_opex[-1]) #€
                else:
                    cost_cash_flow.append(BES_opex[-1])           


        cost_dsctd = [] #€
        for i in range(len(cost_cash_flow)):
            cost_dsctd.append(cost_cash_flow[i]/(1+pyo.value(m.IR))**i)

        dsctd_cash_flows.append(cost_dsctd)

        P_dch_dsctd = [] #MWh
        for i in range(len(cost_cash_flow)):
            P_dch_dsctd.append(P_dch[-1]/(1+pyo.value(m.IR))**i)

        LCOS = sum(cost_dsctd)/sum(P_dch_dsctd) #€/MWh
    else:
        LCOS = float('NaN')

    EM_COST = pyo.value(sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX))
    
    data = pd.DataFrame({'Er_BES [MWh]': Er_BES,
                     'Pr_BES [MW]': Pr_BES,
                     'Pr_diesel [MW]': Pr_dg,
                     'BES cost [million euros]': BES_capex[-1]/1e6 + BES_opex[-1]/1e6 ,
                     'Fuel cost [million euros]': dg_opex[-1]/1e6,
                     'LCOS [€/MWh]': LCOS,
                     'Fuel Consumption [L]': EM_COST
                     })
    
    data_time = pd.DataFrame({'Datetime': design.P_ren['Datetime'],
                              'SOC': SOC[-1],
                              'P_BES': P_BES[-1],
                              'P_curt': P_curt[-1],
                              'P_dg': P_dg[-1],
                              'P_prod':P_prod,
                              'P_load': P_load

                     })
    return data, data_time
 

''' Reading the data from csv files'''
P_load_data = pd.read_excel('load_data.xlsx', sheet_name='Yearly Load', header=0) #MW
P_ren_read = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')



load_avg = np.linspace(min(P_ren_read['Power']),max(P_ren_read['Power']), 20) # 5% jumps from min to max ren power

load = load_avg[1]*np.ones(8760)/1e6 #MW
P_load_data['Load [MW]'] = load 

design = microgrid(Pr_BES=20,Er_BES=250, P_load=P_load_data, P_ren=P_ren_read)

df = pd.DataFrame(columns=['Load [%]','Er_BES [MWh]','Pr_BES [MW]','Pr_diesel [MW]','BES cost [million euros]',\
                   'Fuel cost [million euros]','LCOS [€/MWh]','Fuel Consumption [L]'])

i = 0
for load in load_avg[1:3]:
    design.P_load['Load [MW]'] = load*np.ones(8760)/1e6 #MW
    data, _ = dispatcher_GA(design, True)

    df = pd.concat([df,data], ignore_index=True)
    df['Load [%]'][i] = (load - load_avg[1])/(load_avg[-1] - load_avg[1])*100
    i += 1 

#this sections calculates how much it is spent if there is no BES with minimum load
load = load_avg[1]
design.P_load['Load [MW]'] = load*np.ones(8760)/1e6 #MW
design.Er_BES = 0
design.Pr_BES = 0
data, _ = dispatcher_GA(design, False)
df = pd.concat([df,data], ignore_index=True)
df['Load [%]'][i] = (load - load_avg[1])/(load_avg[-1] - load_avg[1])*100

df['Total cost [million euros]'] = df['BES cost [million euros]'] + df['Fuel cost [million euros]']

# %%

fig, ax_cost = plt.subplots(figsize=(15,7))
fig.set_dpi(200)

ax_cost.set_xlabel("Load [% max RES]")
ax_cost.set_ylabel("Cost [million €]")
ax_cost.set_title("Costs")

ax_cost.plot(df['Load [%]'], df['BES cost [million euros]'], color= 'black')
ax_cost.plot(df['Load [%]'], df['Fuel cost [million euros]'], color= 'blue')
ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')
ax_cost.legend(['BES', 'Fuel', 'Total'],loc=4)

ax_LCOS = ax_cost.twinx()
ax_LCOS.set_ylabel("LCOS [€/MWh]")

ax_LCOS.plot(df['Load [%]'], df['LCOS [€/MWh]'], color= 'green')
ax_LCOS.legend(['LCOS'])


ax_cost.spines['right'].set_color("black")
ax_LCOS.spines['right'].set_color("green")

plt.grid(True)
plt.show()
plt.close()

# %% TIME SERIES PLOT
plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams.update({'font.size': 12})

display_start = 0
display_end= 24*31

time_horizon = range(display_start, display_end)

fig, ax_pow = plt.subplots()

ax_pow.set_ylabel("Power [MW]")
ax_pow.set_xlabel("Time [h]")
ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_load'][display_start:display_end], color='black')
ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_BES'][display_start:display_end], color = 'blue')
ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_prod'][display_start:display_end], color = 'green')
ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_curt'][display_start:display_end], color='red')
ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_dg'][display_start:display_end], color='orange')
ax_pow.legend(['P_Load', 'P_BES', 'P_RES', 'P_curt','P_diesel'],loc=4)


ax_pow.set_title("Energy dispatch diesel + PV + BESS")

ax_SOC = ax_pow.twinx()  # instantiate a second axes that shares the same x-axis

ax_SOC.set_ylabel('SOC [-]')  # we already handled the x-label with ax1
ax_SOC.plot(data_time['Datetime'][display_start:display_end], data_time['SOC'][display_start:display_end], color='cyan')
ax_SOC.legend(['SOC'],loc=2)

ax_pow.spines['right'].set_position(('axes',0.15))

ax_pow.spines['right'].set_color("black")
ax_SOC.spines['right'].set_color("cyan")



fig.set_dpi(200)
ax_pow.tick_params(axis='both', which='major')
ax_SOC.tick_params(axis='both', which='major')

ax_pow.xaxis.set_major_locator(DayLocator())
ax_pow.xaxis.set_major_formatter(DateFormatter('%m-%d'))

# plt.savefig('powers_SOC.png',bbox_inches='tight', dpi=150)

plt.grid(True)
plt.show()
plt.close()

#%% Plotting the renewable generation
P_ren_read = P_ren_read.set_index('Datetime')
fig = plt.figure(figsize=[25, 7])

ax_pow = fig.add_subplot()

ax_pow.set_xlabel("Month")
ax_pow.set_ylabel("Power [MW]")

ax_pow.plot(P_ren_read['Power']/1e6, color='black')

plt.title("RES generation curve")
plt.xticks(rotation=45)

plt.grid(True)

ax_pow.xaxis.set_major_locator(MonthLocator(bymonthday=15))
ax_pow.xaxis.set_major_formatter(DateFormatter('%B'))

plt.savefig('RES_curve.png',bbox_inches='tight', dpi=150)

# %%

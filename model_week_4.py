import time
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#grid connected microgrid with pv+wind, load, BESS, curtailment and no diesel

''' in this one i was trying to see how much changing the optimization horizon (in terms of months)
would change the results'''

# price = pd.read_excel("Cycle_calculations.xlsx", sheet_name='Electricity Prices', usecols = "A") #eur/MWh
# price_dict = dict()

# for i in range(len(price)):
#     price_dict[i]=price['Price [€/MWh]'].values[i]
    
size_font = 10
months = 3

BES_cost = np.array([])
grid_cost = np.array([])
Er_BES = np.array([])
Pr_BES = np.array([])
code_time = np.array([])
LCOS = np.array([])

price_data = pd.read_csv("PriceCurve_SE3_2021.csv", header = 0, nrows = 8760, sep=';') #eur/MWh
P_load_data = pd.read_excel('load_data.xlsx', sheet_name='Yearly Load', header=0) #MW
P_ren_data = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760) #W
P_ren_data['Power'] = P_ren_data['Power']/1e6*7 #MW

for n_month in range(1,months+1):
    
    optim_time = 8760 # number of hours to display
    time_range_optim = range(optim_time)
    
    price_dict = dict()
    
    for i in time_range_optim:
        price_dict[i] = price_data['Grid_Price'].values[i]
    
    P_load_dict = dict() 
    for i in time_range_optim:
        P_load_dict[i] = P_load_data['Load [MW]'].values[i]
        
    P_ren_dict = dict()
    for i in time_range_optim:
        P_ren_dict[i] = P_ren_data['Power'].values[i] #MW
        
    # ax_pow = plt.gca()
    
    # P_load.iloc[0:len(price)].plot(kind="line", y = 'Load [MW]', ax = ax_pow)
    # P_ren.iloc[0:len(price)].plot(kind="line", y = 'Power', ax = ax_pow)
    
    # plt.show()
    
    ren_surplus = sum(P_ren_data['Power'].values[i] for i in time_range_optim)-sum(P_load_data['Load [MW]'].values[i] for i in time_range_optim)
    # print('The renewable energy surplus is of ',ren_surplus , 'MWh')
    
    
        
    m = ConcreteModel()
    
    m.iIDX = Set(initialize = time_range_optim)
    m.P_load = Param(m.iIDX,initialize=P_load_dict)
    m.P_ren = Param(m.iIDX, initialize=P_ren_dict)
    m.price = Param(m.iIDX, initialize = price_dict)
    m.eff_ch = Param(initialize=sqrt(0.95)) #square root of roundtrip efficiency used in amazing reference
    m.eff_dch = Param(initialize=sqrt(0.95))
    m.lifetime = Param(initialize=10) #years
    m.C_P = Param(initialize=320) #$/kW
    m.C_E = Param(initialize=360) #$/kWh
    m.C_inst = Param(initialize=15) #$/kWh
    m.C_OM = Param(initialize=5) #$/kW
    m.IR = Param(initialize=0.05)
    m.sigma = Param(initialize=0.002/24) #original daily self discharge is 0,2% -> we need an hourly self discharge
    m.gamma_min = Param(initialize=0)
    m.gamma_MAX = Param(initialize=100) #maximum 10 (Stefan)
    m.P_BES_MAX = Param(initialize=5*max(P_load_data['Load [MW]']))
    
    m.P_ch = Var(m.iIDX, domain=NonNegativeReals)
    m.P_dch = Var(m.iIDX, domain = NonNegativeReals)
    m.P_grid = Var(m.iIDX, domain = Reals)
    #m.P_curt = Var(m.iIDX, domain = NonNegativeReals)
    m.Pr_BES = Var(domain=NonNegativeReals, bounds=(0, 10*max(P_load_data['Load [MW]'])))
    m.Er_BES = Var(domain=NonNegativeReals)
    m.SOC = Var(m.iIDX, domain=NonNegativeReals)
    m.SOC_ini = Var(domain=NonNegativeReals)
    m.bin_dch = Var(m.iIDX, domain=Binary)
    
    def obj_funct(m):
        return (m.Pr_BES*m.C_P + m.Er_BES*(m.C_E+m.C_inst))*1e3 + sum((m.P_grid[i]*m.price[i]) for i in m.iIDX)*8760/len(m.iIDX)*m.lifetime #add battery lifetyime
    m.obj = Objective(rule=obj_funct, sense=minimize)
    
    def f_equilibrium(m,i):
        return m.P_grid[i] + m.P_dch[i] - m.P_ch[i] + m.P_ren[i] == m.P_load[i] #+ m.P_curt[i]
    
    m.cstr_eq = Constraint(m.iIDX, rule = f_equilibrium)
    
    def f_SOC_lim(m,i):
        return m.SOC[i]<= m.Er_BES
    
    m.cstr_SOC_lim = Constraint(m.iIDX, rule=f_SOC_lim)
    
    def f_SOC_ini_lim(m):
        return m.SOC_ini <= m.Er_BES
    
    m.cstr_SOC_ini_lim = Constraint(rule=f_SOC_ini_lim)
    
    def f_SOC(m,i):
        if i == 0:
            return m.SOC[i] == m.SOC_ini #add condition on final SOC of simulation
        elif i == len(m.iIDX) - 1:
            return m.SOC[i] == m.SOC_ini
        else:
            return m.SOC[i] == m.SOC[i-1]*(1-m.sigma) + m.P_ch[i]*m.eff_ch - m.P_dch[i]/m.eff_dch
    
    m.cstr_SOC = Constraint(m.iIDX, rule = f_SOC)
    
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
    
    start = time.time()
    opt = SolverFactory("gurobi")
    opt.solve(m)
    code_time = np.append(code_time, time.time() - start)
    
    display_time = 24*4 # number of hours to display
    time_horizon = range(display_time)
    
    SOC = np.array([value(m.SOC[i]) for i in m.iIDX])
    SOC = SOC/value(m.Er_BES)
    P_ren = np.array([value(m.P_ren[i]) for i in m.iIDX])
    P_BES = np.array([(value(m.P_dch[i]) - value(m.P_ch[i]) ) for i in m.iIDX])
    P_load = np.array([value(m.P_load[i]) for i in m.iIDX])
    P_grid = np.array([value(m.P_grid[i]) for i in m.iIDX])
    P_dch = value(sum(m.P_dch[i] for i in m.iIDX))
    price = np.array([value(m.price[i]) for i in m.iIDX])
    BES_cost = np.append(BES_cost,value((m.Pr_BES*m.C_P + m.Er_BES*(m.C_E+m.C_inst))*1e3)/1e6) #milions
    grid_cost = np.append(grid_cost, value(sum((m.P_grid[i]*m.price[i]) for i in m.iIDX)*8760/len(m.iIDX)*m.lifetime)/1e6) #milions in lifetime
    Er_BES = np.append(Er_BES, value(m.Er_BES))
    Pr_BES = np.append(Pr_BES, value(m.Pr_BES))
    
    LCOS = np.append(LCOS, (BES_cost[n_month-1] + sum((Er_BES[n_month-1]*value(m.C_OM)*1e3/(1+value(m.IR))**(y-1)) 
                                          for y in range(1,value(m.lifetime)+1)))/ \
                sum((P_dch/(1+value(m.IR))**(y-1)) for y in range(1,value(m.lifetime)+1)))
    
    # print('The number of months chosen for optimization is', n_month)
    # print('The code took ', deltaT,' s.')
    # print('The optimal storage capacity is ',Er_BES,' MWh, with a rated power of ', Pr_BES,'MW.')
    # print('The total cost of the battery system is ', BES_cost, 'millions.\n')
    #print('The price payed to withdraw electricity is', grid_cost, 'millions. \n')

data = pd.DataFrame({'Optim. Months': range(1,months+1),
                     'BES cost': BES_cost,
                     'Er_BES': Er_BES,
                     'Pr_BES': Pr_BES,
                     'Grid cost':grid_cost,
                     'LCOS': LCOS
    })

print(data)
# fig, ax_pow = plt.subplots(figsize=(10,8))

# ax_pow.set_ylabel("Power [MW]",fontsize=size_font)
# ax_pow.set_xlabel("time_horizon [h]",fontsize=size_font)
# ax_pow.plot(time_horizon, P_load[0:display_time], color='black')
# ax_pow.plot(time_horizon, P_BES[0:display_time], color = 'blue')
# ax_pow.plot(time_horizon, P_ren[0:display_time], color = 'green')
# ax_pow.plot(time_horizon, P_grid[0:display_time], color='red')
# ax_pow.legend(['P_Load', 'P_BES', 'P_ren', 'P_grid'],loc=4)

# ax_price = ax_pow.twinx()

# ax_price.set_ylabel("Price [€/MWh]",fontsize=size_font)
# ax_price.set_xlabel("Time [h]",fontsize=size_font)
# ax_price.plot(time_horizon, price[0:display_time], color = 'darkorange')
# ax_price.legend(['Electricity Price'],loc=2)



# ax_pow.spines['right'].set_color("black")
# ax_price.spines['right'].set_color("darkorange")


# plt.grid(True)

# fig.set_size_inches(10,8)
# fig.set_dpi(200)
# ax_pow.tick_params(axis='both', which='major', labelsize=size_font)
# ax_price.tick_params(axis='both', which='major', labelsize=size_font)

# plt.savefig('powers_price.png',bbox_inches='tight', dpi=150)

# plt.show()
# plt.close()

# fig, ax_pow = plt.subplots(figsize=(10,8))

# ax_pow.set_ylabel("Power [MW]",fontsize=size_font)
# ax_pow.set_xlabel("Time [h]",fontsize=size_font)
# ax_pow.plot(time_horizon, P_load[0:display_time], color='black')
# ax_pow.plot(time_horizon, P_BES[0:display_time], color = 'blue')
# ax_pow.plot(time_horizon, P_ren[0:display_time], color = 'green')
# ax_pow.plot(time_horizon, P_grid[0:display_time], color='red')
# ax_pow.legend(['P_Load', 'P_BES', 'P_ren', 'P_grid'],loc=4)


# ax_SOC = ax_pow.twinx()  # instantiate a second axes that shares the same x-axis

# ax_SOC.set_ylabel('SOC [-]',fontsize=size_font)  # we already handled the x-label with ax1
# ax_SOC.plot(time_horizon, SOC[0:display_time], color='cyan')
# ax_SOC.legend(['SOC'],loc=2)

# ax_pow.spines['right'].set_position(('axes',0.15))

# ax_pow.spines['right'].set_color("black")
# ax_SOC.spines['right'].set_color("cyan")

# plt.grid(True)


# fig.set_size_inches(10,8)
# fig.set_dpi(200)
# ax_pow.tick_params(axis='both', which='major', labelsize=size_font)
# ax_SOC.tick_params(axis='both', which='major', labelsize=size_font)

# plt.savefig('powers_SOC.png',bbox_inches='tight', dpi=150)

# plt.show()
# plt.close()

# fig, ax_pow = plt.subplots(figsize=(10,8))

# ax_pow.set_ylabel("Energy throughput [%]",fontsize=size_font)
# ax_pow.set_xlabel("Time [h]",fontsize=size_font)
# ax_pow.set_title("Energy throughput")
# ax_pow.plot(time_horizon, abs(P_BES[0:display_time])/Er_BES*100, color = 'blue')




# plt.grid(True)


# fig.set_size_inches(10,8)
# fig.set_dpi(200)
# ax_pow.tick_params(axis='both', which='major', labelsize=size_font)
# # ax_SOC.tick_params(axis='both', which='major', labelsize=size_font)

# # plt.savefig('powers_SOC.png',bbox_inches='tight', dpi=150)

# plt.show()
# plt.close()





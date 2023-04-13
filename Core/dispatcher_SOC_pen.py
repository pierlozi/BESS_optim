import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from math import sqrt
import time
import numpy as np
import pandas as pd

def MyFun(design, bin_var): #bin_var is to tell if power and energy rating are variables (bin_var=True) or input parameters (bin_var=False) for the dispatcher
    
    print("Started.")
    optim_horiz = design.optim_horiz # number of hours to optmize (optmization horizon)
    time_range_optim = range(optim_horiz)
    
    '''Initializing the lists of all the paramters and variables which values I want to store'''
    code_time = []

    Er_BES = []
    Pr_BES = []

    SOC = []
    P_BES = []
    E_dch = []
    P_dch = []
    P_ch = []
    P_dg = []
    P_curt = []

    BES_capex = [] #installation and replacement cost
    BES_opex = [] #operation and maintenance cost

    P_thr = []

    dg_capex = []
    dg_opex = []    
    Pr_dg = []

    BES_cyclelife = []
    cycles_y = []
    dsctd_cash_flows = []

    '''I multiply the renewable power production with a scaling factor to bring it to the same order of magnitutde of the load'''
    P_prod_data = design.P_ren['Power']*design.RES_fac/1e6 #MW 

    '''I produce dictionaries of the data imported as the pyomo framework reads input parameters in form of dictionary'''
    P_load_dict = dict() 
    for i in time_range_optim:
        P_load_dict[i] = design.P_load['Load [MW]'].values[i]
        
    P_prod_dict = dict()
    for i in time_range_optim:
        P_prod_dict[i] = P_prod_data.iloc[i] #MW


    '''Here I set the max power of the BESS'''
    P_BES_MAX = 10*max(max(design.P_load['Load [MW]']),max(P_prod_data))

    ''' bess_bin is the variable that tells which type of microgrid we have, 0 = only diesel ; 1 = diesel + res; 2 = diesel + res + storage.
    I use it so that the model compiles different objective functions and constraints for the two types.'''


    m = pyo.ConcreteModel()

    #m.iIDX is the set which keeps the time in the simulation
    m.iIDX = pyo.Set(initialize = time_range_optim)
    m.BES_life = pyo.Set(initialize = range(design.floatlife))

    '''importing data in the pyomo framewrok''' 
    m.P_load = pyo.Param(m.iIDX,initialize=P_load_dict)
    m.P_prod = pyo.Param(m.iIDX, initialize=P_prod_dict)
    #m.price = pyo.Param(m.iIDX, initialize = price_dict)

    '''initializing parameters for the simulation'''

    #the charging and dischargin efficiencies are calculeted using a roundrtip efficiency value
    #of 0.95 (Reference: Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year)
    m.eff_ch = pyo.Param(initialize=sqrt(design.eff)) 
    m.eff_dch = pyo.Param(initialize=sqrt(design.eff))


    m.floatlife = pyo.Param(initialize = design.floatlife) #years
    m.C_P = pyo.Param(initialize= design.C_P) #$/kW
    m.C_E = pyo.Param(initialize= design.C_E) #$/kWh
    m.C_inst = pyo.Param(initialize= design.C_inst) #$/kWh (the reference doesnt take into account installation)
    m.C_POM = pyo.Param(initialize= design.C_POM) #$/kW operation cost related to power
    m.C_EOM = pyo.Param(initialize= design.C_EOM) #$/kWh operation cost related to energy
    m.sigma = pyo.Param(initialize= design.sigma/24) #original daily self discharge is 0,2% -> we need an hourly self discharge
    m.IR = pyo.Param(initialize = design.IR)

    # '''Adding the table of DoD - cycle life to implement battery degradation'''
    
    m.DoD = pyo.Param(initialize = design.DoD)
    m.cyclelife = pyo.Param(initialize = design.cyclelife)

    m.gamma_min = pyo.Param(initialize=0)
    m.gamma_MAX = pyo.Param(initialize=100000) #maximum 10 (Stefan) -> for now I leave it very high to see how the system behaves

    m.P_BES_MAX = pyo.Param(initialize=P_BES_MAX)

    m.price_f = pyo.Param(initialize= design.price_f)
    m.C_DG = pyo.Param(initialize= design.C_DG)

    #empirical parameters for diesel fuel consumption from 
    # "Multi objective particle swarm optimization of hybrid micro-grid system: A case study in Sweden"
    m.alpha = pyo.Param(initialize=0.24) #L/kW
    m.beta = pyo.Param(initialize=0.084) #L/kW

    #minimum up and down time for diesel from
    # "Optimal sizing of battery energy storage systems in off-grid micro grids using convex optimization"
    m.UT = pyo.Param(initialize = 5) #h
    m.DT = pyo.Param(initialize = 1) #h

    #max and min power rating of the diesel generator, I choose it arbitrarily
    m.Pr_dg_MAX = pyo.Param(initialize = max(design.P_load['Load [MW]']))
    m.Pr_dg_MIN = pyo.Param(initialize = 0.1*sum(design.P_load['Load [MW]'])/len(design.P_load['Load [MW]']))


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
        def obj_funct(m): 
            return sum((100 - m.SOC[i]) for i in m.iIDX)*design.SOC_w*8760/len(m.iIDX)+\
                   (m.Pr_BES*(m.C_P + m.C_POM*sum((1/(1+m.IR)**y) for y in m.BES_life)) +\
                    m.Er_BES*(m.C_E + m.C_inst + m.C_EOM*sum((1/(1+m.IR)**y) for y in m.BES_life)))*1e3 +\
                    m.Pr_dg*m.C_DG*1e3 +\
                   (m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX))*8760/len(m.iIDX)*sum((1/(1+m.IR)**y) for y in m.BES_life)
    
    else:
        def obj_funct(m):
            return m.Pr_dg*m.C_DG*1e3 +\
                  (m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX))*8760/len(m.iIDX)*sum((1/(1+m.IR)**y) for y in m.BES_life)
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

    def f_dg_lim_dwn(m,i):
        return m.P_dg[i] >= 0.2*m.Pr_dg

    m.cstr_dg_lim_dwn = pyo.Constraint(m.iIDX, rule=f_dg_lim_dwn)

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

    if pyo.value(m.Er_BES) != 0:
        SOC.append(np.array([pyo.value(m.SOC[i]) for i in m.iIDX])/pyo.value(m.Er_BES))
    else:
        SOC.append(np.array([pyo.value(m.SOC[i]) for i in m.iIDX]))

    P_BES.append(np.array([(pyo.value(m.P_dch[i]) - pyo.value(m.P_ch[i]) ) for i in m.iIDX]))
    E_dch.append(pyo.value(sum(m.P_dch[i] for i in m.iIDX)))
    P_dch.append(np.array([pyo.value(m.P_dch[i]) for i in m.iIDX]))
    P_ch.append(np.array([pyo.value(m.P_ch[i]) for i in m.iIDX]))
    BES_capex.append(pyo.value((m.Pr_BES*m.C_P + m.Er_BES*(m.C_E+m.C_inst))*1e3)) #€
    BES_opex.append(pyo.value(m.Pr_BES*m.C_POM*1e3 + m.Er_BES*m.C_EOM*1e3)) #€/year
    Er_BES.append(pyo.value(m.Er_BES))
    Pr_BES.append(pyo.value(m.Pr_BES))

    # chi = np.array([value(m.chi[b]) for b in m.bIDX])
    # LEr = np.array([value(m.LEr[b]) for b in m.bIDX])
        
    P_curt.append(np.array([pyo.value(m.P_curt[i]) for i in m.iIDX]))

    P_dg.append(np.array([pyo.value(m.P_dg[i]) for i in m.iIDX]))
    dg_capex.append(pyo.value(m.Pr_dg*m.C_DG*1e3)) #€
    dg_opex.append(pyo.value((m.price_f*sum((m.alpha*m.Pr_dg + m.beta*m.P_dg[i])*1e3 for i in m.iIDX))*8760/optim_horiz)) #€/year   
    Pr_dg.append(pyo.value(m.Pr_dg))
    # u_dg = np.array([pyo.value(m.u_dg[i]) for i in m.iIDX])
    # w_dg = np.array([pyo.value(m.w_dg[i]) for i in m.iIDX])
    # v_dg = np.array([pyo.value(m.v_dg[i]) for i in m.iIDX])

    P_thr.append(abs(P_BES[-1]))

    if pyo.value(m.Er_BES) != 0 and pyo.value(m.Pr_BES) != 0: #to avoid division by zero
        cycles_y.append(sum(P_thr[-1])/Er_BES[-1])

        BES_cyclelife.append(round(pyo.value(m.cyclelife)/cycles_y[-1]))

        cost_cash_flow  = []
        for i in range(0, design.mine_life):
            if design.mine_life > pyo.value(m.floatlife): #if the mine has a life longer than shelflife of battery
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
            P_dch_dsctd.append(E_dch[-1]/(1+pyo.value(m.IR))**i)

        LCOS = sum(cost_dsctd)/sum(P_dch_dsctd) #€/MWh
    else:
        LCOS = float('NaN')

    
    data = pd.DataFrame({'Er_BES [MWh]': Er_BES,
                     'Pr_BES [MW]': Pr_BES,
                     'Pr_diesel [MW]': Pr_dg,
                     'BES cost [million euros]': BES_capex[-1]/1e6 + BES_opex[-1]/1e6 ,
                     'DG cost [million euros]': dg_capex[-1]/1e6,
                     'LCOS [€/MWh]': LCOS,
                     'Fuel Cost [million euros]': dg_opex[-1]/1e6,
                     'Total Cost [million euros]': BES_capex[-1]/1e6 + BES_opex[-1]/1e6 + dg_capex[-1]/1e6 + dg_opex[-1]/1e6,
                     'Lifetime cost [million euros]': pyo.value(m.obj-sum((100 - m.SOC[i]) for i in m.iIDX))/1e6 
                     })
    
    data_time = pd.DataFrame({'Datetime': design.P_ren['Datetime'][0:optim_horiz],
                              'SOC': SOC[-1],
                              'P_ch': P_ch[-1],
                              'P_dch': P_dch[-1],
                              'P_BES': P_BES[-1],
                              'P_curt': P_curt[-1],
                              'P_dg': P_dg[-1],
                              'P_prod':P_prod,
                              'P_load': P_load

                     })
    
    print("Done.")
    return data, data_time
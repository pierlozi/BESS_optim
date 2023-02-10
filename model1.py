from pyomo.environ import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
size_font = 10

P_load = pd.read_excel('load_data.xlsx', sheet_name='Yearly Load', header=0)

P_load_dict = dict()
for i in range(len(P_load)):
    P_load_dict[i] = P_load['Load [MW]'].values[i]


P_ren = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760)
P_ren['Power'] = P_ren['Power']/1e6*10 #MW

P_ren_dict = dict()
for i in range(len(P_ren)):
    P_ren_dict[i] = P_ren['Power'].values[i] #MW

model = ConcreteModel()
model.hIDX = Set(initialize = range(len(P_ren_dict))) 
model.P_load = Param(model.hIDX, initialize = P_load_dict) #kW
model.P_ren = Param(model.hIDX, initialize = P_ren_dict) #kW
model.eff_ch = Param(initialize=sqrt(0.95)) #square root of roundtrip efficiency used in amazing reference
model.eff_dch = Param(initialize=sqrt(0.95))
model.C_P = Param(initialize=320) #$/kW
model.C_E = Param(initialize=360) #$/kWh
model.C_inst = Param(initialize=15) #$/kWh
model.C_OM = Param(initialize=5) #$/kW
model.IR = Param(initialize=0.05)
model.sigma = Param(initialize=0.002/24) #original daily self discharge is 0,2% -> we need an hourly self discharge
model.life = Param(initialize=15)
model.gamma_min = Param(initialize=0)
model.gamma_MAX = Param(initialize=13)
model.price_g = Param(initialize = 0.5) #$/kWh

model.Pr_BES = Var(domain=NonNegativeReals)
model.Er_BES = Var(domain=NonNegativeReals)
model.P_grid = Var(model.hIDX, domain=NonNegativeReals)
model.P_BES = Var(model.hIDX, domain=Reals)
model.P_ch = Var(model.hIDX, domain=NonNegativeReals)
model.P_dch = Var(model.hIDX, domain=NonNegativeReals)
model.SOC = Var(model.hIDX, domain=NonNegativeReals)
model.SOC_ini = Var(domain=NonNegativeReals)
model.chi_dch = Var(model.hIDX,domain=Binary)


def obj_expression(model):
  return sum(model.P_grid[h] for h in model.hIDX)

model.obj = Objective(rule=obj_expression, sense=minimize)

def equilibrium_constraint(model,h):
    return model.P_BES[h] + model.P_ren[h] + model.P_grid[h] == model.P_load[i]

model.eq_cstr = Constraint(model.hIDX, rule=equilibrium_constraint)

def power_BES_constraint(model,h):
    return model.P_BES[h] == model.P_dch[h] - model.P_ch[h]

model.P_BES_str = Constraint(model.hIDX, rule=power_BES_constraint)

def SOC_ini_constraint(model):
    return model.SOC_ini <= model.Er_BES

model.SOC_ini_cstr = Constraint(rule=SOC_ini_constraint)

def SOC_bounds_constraint(model,h):
    return model.SOC[h] <= model.Er_BES

model.SOC_bounds_cstr = Constraint(model.hIDX, rule=SOC_bounds_constraint)

def SOC_constraint(model,h):
    if h==0:
        return model.SOC[h] == model.SOC_ini
    else:
        return model.SOC[h] == model.SOC[h-1]*(1-model.sigma) - model.P_dch[h]/model.eff_dch + model.P_ch[h]*model.eff_ch

model.SOC_cstr = Constraint(model.hIDX, rule=SOC_constraint)


def capacity_constraint_lower(model):
    return model.Er_BES>=model.gamma_min*model.Pr_BES 

model.gamma_low_cstr = Constraint(rule=capacity_constraint_lower)

def capacity_constraint_upper(model):
    return model.Er_BES<=model.gamma_MAX*model.Pr_BES 

model.gamma_up_cstr = Constraint(rule=capacity_constraint_upper)

def discharge_constraint_1 (model,h):
    return model.P_dch[h] <= model.Pr_BES

model.dch_cstr_1 = Constraint(model.hIDX, rule=discharge_constraint_1)

def discharge_constraint_2 (model,h):
    return model.P_dch[h] <= model.chi_dch[h]*1e40

model.dch_cstr_2 = Constraint(model.hIDX, rule=discharge_constraint_2)

def charge_constraint_1 (model,h):
    return model.P_ch[h] <= model.Pr_BES

model.ch_cstr_1 = Constraint(model.hIDX, rule=charge_constraint_1)

def charge_constraint_2 (model,h):
    return model.P_dch[h] <= (1-model.chi_dch[h])*1e40

model.ch_cstr_2 = Constraint(model.hIDX, rule=charge_constraint_2)

solver = SolverFactory("cbc.exe")
result = solver.solve(model)

value(model.Er_BES)
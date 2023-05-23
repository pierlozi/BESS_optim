
#%% parameters
# data from 'Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year'
# C_P = 320 #$/kW
# C_E = 360 #$/kWh
# C_inst = 15 #$/kWh
# C_POM = 5 #$/kW operation cost related to power
# C_EOM = 0 #$/kWh operation cost related to energy
# sigma = 0,2/100 #original daily self discharge is 0,2% -> we need an hourly self discharge
# m.IR = 5/100
# floatlife = 10

#data from 'Projecting the Future Levelized Cost of Electricity Storage Technologies'
# C_P = 678 #$/kW
# C_E = 802 #$/kWh
# C_inst = 0 #$/kWh (the reference doesnt take into account installation)
# C_POM = 10 #$/kW operation cost related to power
# C_EOM = 3 #$/kWh operation cost related to energy
# sigma = 0 #original daily self discharge is 0,2% -> we need an hourly self discharge
# IR = 8/100

#data from Luka
# C_P = 160 #$/kW
# C_E = 180 #$/kWh
# C_inst = 0 #$/kWh (the reference doesnt take into account installation)
# C_POM = 0 #$/kW operation cost related to power
# C_EOM = 0.125 #$/kWh operation cost related to energy
# sigma = 0 #original daily self discharge is 0,2% -> we need an hourly self discharge
# IR = 5/100

# mine_life = 13 #years # randomly chosen 

# price_f = 1.66 # €/L
# C_DG = 600 #€/kW

 #empirical parameters for diesel fuel consumption from 
# "Multi objective particle swarm optimization of hybrid micro-grid system: A case study in Sweden"
# alpha = 0.24
# beta = 0.084

# "Multi-objective design of PV– wind– diesel– hydrogen– battery systems"
# alpha = 0.246
# beta = 0.08145 

# cyclelifes = [170000, 48000, 21050, 11400, 6400, 4150, 3500, 3000, 2700, 2500]
# DoDs = [10, 20, 30, 40, 50, 60, 65, 70, 75, 80]

# data from 'Optimal sizing of battery energy storage systems in off-grid micro grids using convex optimization'
# p_NOx = 10.0714
# p_SO2 = 2.3747
# p_CO2 = 0.0336
# ef_NOx = 0.0218
# ef_SO2 = 0.000454
# ef_CO2 = 0.001432

#%% class
import pandas as pd
# class to build object with all the design parameters to pass to the optimizer 
class MG(): #MG stands for microgrid

    def __init__(self, optim_horiz = 8760, Er_BES=None, Pr_BES=None, P_load=pd.DataFrame() ,\
                 P_ren=pd.DataFrame(), mine_life=13, RES_fac=1, floatlife=10, C_P=360, C_E=320, \
                 C_inst=15, C_POM=5, C_EOM=0, sigma=0.2/100, IR=5/100, DoD=75, cyclelife=2700, \
                 eff = 0.95, price_f=1.66, C_DG=500, SOC_w = 0, alpha = 0.246, beta = 0.08145, \
                 p_NOx = 10.0714, p_SO2 = 2.3747, p_CO2 = 0.0336,\
                 ef_NOx = 0.0218, ef_SO2 = 0.000454, ef_CO2 = 0.001432 ):
        
        self.optim_horiz = optim_horiz
        
        #when the dispatcher is run inside the GA Er_BES and Pr_BES are input parameters
        self.Er_BES = Er_BES  # [MWh] capacity rating of the BES
        self.Pr_BES = Pr_BES  # [MW] power rating of the BES

        self.SOC_w = SOC_w    #[-] the weight given to the SOC inclusion in the objective function

        self.P_load = P_load  # [MW] dictionary P_load['Load [MW]'] with the yearly load profile 
        self.P_ren = P_ren    # [W] dictionary P_ren['Power'] with yearly RES generation
        self.RES_fac = RES_fac # [-] multiplying factor to increase/decrease the RES generation
        self.mine_life = mine_life #[y] expected life of the mine

        self.floatlife = floatlife # [y] expected floatlife of BES !!!IMPORTANT!!! very influencial factor in the optimization
        self.C_P = C_P # [$/kW] power rating CAPEX
        self.C_E = C_E # [$/kWh] energy rating CAPEX
        self.C_inst = C_inst # [$/kWh] installation cost -> CAPEX
        self.C_POM = C_POM # [$/kW] power rating OPEX
        self.C_EOM = C_EOM # [$/MMh] energy rating OPEX
        self.sigma = sigma # [% capacity/day] daily self discharge
        self.IR = IR       # [%] interes rate

        self.DoD = DoD # [%] Depth of Discharge at which battery works
        self.cyclelife = cyclelife # [cycles] cyclelife corresponding to set DoD
        self.eff = eff # battery charge and discharge efficiency

        self.price_f = price_f # [€/L] price of diesel
        self.C_DG = C_DG #[€/kW] DG CAPEX
        self.alpha = alpha #factor for fuel consumption
        self.beta = beta #factor for fuel consumption

        self.p_NOx = p_NOx #€/kg
        self.p_SO2 = p_SO2 #€/kg
        self.p_CO2 = p_CO2 #€/kg
        self.ef_NOx = ef_NOx #kg/kWh
        self.ef_SO2 = ef_SO2 #kg/kWh
        self.ef_CO2 = ef_CO2 #kg/kWh

# %%

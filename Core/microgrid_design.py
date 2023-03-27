
#%% parameters
# data from 'Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year'
# C_P = 320 #$/kW
# C_E = 360 #$/kWh
# C_inst = 15 #$/kWh
# C_POM = 5 #$/kW operation cost related to power
# C_EOM = 0 #$/kWh operation cost related to energy
# sigma = 0,2/100 #original daily self discharge is 0,2% -> we need an hourly self discharge
# m.IR = 5/100

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

# floatlife = 9 #years
# mine_life = 13 #years

# price_f = 1.66 # €/L
# C_DG = 600 #€/kW

# DoD = 75 # %
# cyclelife = 2700 #cycles

#%% class
import pandas as pd
# class to build object with all the design parameters to pass to the optimizer 
class MG(): #MG stands for microgrid

    def __init__(self, Er_BES=None, Pr_BES=None, P_load=pd.DataFrame() , P_ren=pd.DataFrame(), mine_life=13, RES_fac=7, floatlife=9, C_P=160, C_E=180, C_inst=0, C_POM=0, C_EOM=0, sigma=0.2/100, IR=5/100, DoD=75, cyclelife=2700, price_f=1.66, C_DG=600):
        #when the dispatcher is run inside the GA Er_BES and Pr_BES are input parameters
        self.Er_BES = Er_BES  # [MWh] capacity rating of the BES
        self.Pr_BES = Pr_BES  # [MW] power rating of the BES

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

        self.price_f = price_f # [€/L] price of diesel
        self.C_DG = C_DG
# %%
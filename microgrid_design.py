
# class to build object with all the design parameters to pass to the optimizer 
class MG(): #MG stands for microgrid

    def __init__(self, Er_BES, Pr_BES, P_load, P_ren, mine_life, RES_fac, floatlife, C_P, C_E, C_inst, C_POM, C_EOM, sigma,IR, DoD, cyclelife, price_f):
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
import math

def MyFun(design, E_dch, res_val_bin : bool): #cyclelife from raifnlow, the others are given

    Er_BES = design.Er_BES
    Pr_BES = design.Pr_BES
    cyclelife = design.cyclelife
    minelife = design.minelife
    floatlife = design.floatlife
    DR = design.DR

    BES_CAPEX = design.Pr_BES*design.C_P + design.Er_BES*(design.C_E+design.C_inst)*1e3 #€
    BES_OPEX = design.Pr_BES*design.C_POM*1e3 + design.Er_BES*design.C_EOM*1e3 #€/year

    DG_CAPEX = design.DG_CAPEX*1e6 #€
    DG_OPEX = design.DG_OPEX*1e6 #€

    #here I compute how many years of life the battery still has when the project is over (minelife), to compute residual value. I take into
    # account also that the battery will have been changed

    if res_val_bin:

        if minelife>=floatlife: #if the life of the mine is greater then the maximum life of the battery

            if floatlife >= cyclelife:
                BES_usage, _ = math.modf(minelife/cyclelife)
            else:
                BES_usage = (minelife%floatlife)/cyclelife
        else:
            BES_usage, _ = math.modf(minelife/cyclelife)

        res_val = BES_CAPEX * (1 - BES_usage)
    else:
        res_val = 0

    if Er_BES.any() != 0 and Pr_BES.any() != 0: #to avoid division by zero
        
        
        LCOS_cost_cash_flow  = []
        tot_cost_cash_flow = []
        for i in range(0, minelife):

            if i == 0:

                LCOS_cost_cash_flow.append(BES_CAPEX + BES_OPEX)  #€
                tot_cost_cash_flow.append(BES_CAPEX + BES_OPEX + DG_CAPEX + DG_OPEX)
            
            else:
                
                if minelife > floatlife: #if the mine has a life longer than floatlife of battery, it either has to be replaced at floatlife or cyclelife
                    
                    if cyclelife >= floatlife: #battery has to be changed at floatlife
                        
                        if i == floatlife - 1:    
                            LCOS_cost_cash_flow.append(BES_CAPEX + BES_OPEX)  #€
                        else:
                            LCOS_cost_cash_flow.append(BES_OPEX)  #€

                    else: #the battery has to be changed at cyclelife

                        if i == cyclelife - 1:    
                            LCOS_cost_cash_flow.append(BES_CAPEX + BES_OPEX)  #€
                        else:
                            LCOS_cost_cash_flow.append(BES_OPEX)  #€

                else: #if the battery has a longer floatlife life than the minelife, it either has not to be changed or to be changed at cyclelife
                    
                    if cyclelife > minelife: # the battery does not need to be replaced

                        LCOS_cost_cash_flow.append(BES_OPEX)

                    else: #the battery needs to be changed at cyclelife

                        if i == cyclelife - 1:    
                            LCOS_cost_cash_flow.append(BES_CAPEX + BES_OPEX)  #€
                        else:
                            LCOS_cost_cash_flow.append(BES_OPEX)  #€   

                tot_cost_cash_flow.append(LCOS_cost_cash_flow[-1] + DG_OPEX) 

        LCOS_cost_cash_flow[-1] = LCOS_cost_cash_flow[-1] - res_val
        tot_cost_cash_flow[-1] = tot_cost_cash_flow[-1] - res_val


        LCOS_cost_dsctd = [] #€
        tot_cost_dsctd = []
        for i in range(len(LCOS_cost_cash_flow)):
            LCOS_cost_dsctd.append(LCOS_cost_cash_flow[i]/(1 + DR)**i)
            tot_cost_dsctd.append(tot_cost_cash_flow[i]/(1 + DR)**i)

        P_dch_dsctd = [] #MWh
        for i in range(len(LCOS_cost_cash_flow)):
            P_dch_dsctd.append(E_dch/(1 + DR)**i)

        LCOS = sum(LCOS_cost_dsctd)/sum(P_dch_dsctd) #€/MWh

        NPC = sum(tot_cost_dsctd)/1e6 #million €

        DG_ratio = 1 - sum(LCOS_cost_dsctd)/sum(tot_cost_dsctd)
    else:

        LCOS = float('NaN')

        tot_cost_cash_flow = []
        for i in range(0, minelife):

            if i == 0:

                tot_cost_cash_flow.append(DG_CAPEX + DG_OPEX)
            
            else:

                tot_cost_cash_flow.append(DG_OPEX) 


        tot_cost_dsctd = []

        for i in range(len(tot_cost_cash_flow)):
            tot_cost_dsctd.append(tot_cost_cash_flow[i]/(1 + DR)**i)

        NPC = sum(tot_cost_dsctd)/1e6 #million €

        DG_ratio = 1
        
    
    return LCOS, BES_usage, NPC, DG_ratio
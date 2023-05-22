import math

def MyFun(Er: float , Pr: float , cyclelife, minelife, floatlife, DR, capex, opex, E_dch): #cyclelife from raifnlow, the others are given

    #here I compute how many years of life the battery still has when the project is over (minelife), to compute residual value. I take into
    # account also that the battery will have been changed

    if minelife>=floatlife:

        if floatlife >= cyclelife:
            BES_usage, _ = math.modf(minelife, minelife/cyclelife)
        else:
            BES_usage = (minelife%floatlife)/cyclelife
    else:
        BES_usage, _ = math.modf(minelife, minelife/cyclelife)

    res_val = capex * (1 - BES_usage)

    if Er != 0 and Pr != 0: #to avoid division by zero
        
        
        cost_cash_flow  = []
        for i in range(0, minelife):

            if minelife > floatlife: #if the mine has a life longer than floatlife of battery, it either has to be replaced at floatlife or cyclelife
                if cyclelife >= floatlife: #battery has to be changed at floatlife

                    if i == 0:
                        cost_cash_flow.append(capex + opex) # + dg_opex[-1]) #€
                    elif i == floatlife - 1:    
                        cost_cash_flow.append(capex + opex) # + dg_opex[-1]) #€
                    else:
                        cost_cash_flow.append(opex) # + dg_opex[-1]) #€

                else: #the battery has to be changed at cyclelife

                    if i == 0:
                        cost_cash_flow.append(capex + opex) # + dg_opex[-1]) #€
                    elif i == cyclelife - 1:    
                        cost_cash_flow.append(capex + opex) # + dg_opex[-1]) #€
                    else:
                        cost_cash_flow.append(opex) # + dg_opex[-1]) #€

            else: #if the battery has a longer floatlife life than the minelife, it either has nto to be changed or to be changed at cyclelife
                if cyclelife > minelife: # the battery does not need to be replaced

                    if i == 0:
                        cost_cash_flow.append(capex + opex) # + dg_opex[-1]) #€
                    else:
                        cost_cash_flow.append(opex)

                else: #the battery needs to be changed at cyclelife

                    if i == 0:
                        cost_cash_flow.append(capex + opex) # + dg_opex[-1]) #€
                    elif i == cyclelife - 1:    
                        cost_cash_flow.append(capex + opex) # + dg_opex[-1]) #€
                    else:
                        cost_cash_flow.append(opex) # + dg_opex[-1]) #€   

        cost_cash_flow[-1] = cost_cash_flow[-1] - res_val

        cost_dsctd = [] #€
        for i in range(len(cost_cash_flow)):
            cost_dsctd.append(cost_cash_flow[i]/(1 + DR)**i)


        P_dch_dsctd = [] #MWh
        for i in range(len(cost_cash_flow)):
            P_dch_dsctd.append(E_dch/(1 + DR)**i)

        LCOS = sum(cost_dsctd)/sum(P_dch_dsctd) #€/MWh
    else:
        LCOS = float('NaN')
    
    return LCOS, res_val
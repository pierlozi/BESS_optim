import numpy as np
import pandas as pd

import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_dsctd, microgrid_design
from Core import rain_deg_funct, LCOS_funct

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

design = microgrid_design.MG(Pr_BES=17.7, \
                Er_BES=173, \
                DoD = 65,\
                P_load=P_load, \
                P_ren=P_ren_read, \
                optim_horiz= 24
                )

def MyFun(design, Delta): #desgign has to have Er/Pr/DoD _0
    
    design_0 = design

    data, data_time = dispatcher_dsctd.MyFun(design_0, False)

    design.cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)

    F1_0, _ , _ = LCOS_funct.MyFun(design, \
                            capex = data['BES CAPEX [million euros]'].values[0]*1e6, \
                            opex = data['BES OPEX [million euros]'].values[0]*1e6,\
                            E_dch = sum(data_time['P_dch']),\
                            res_val_bin = True
                            )

    F2_0 = data['Emissions Cost [million euros]'].values[0]

    X = np.array([design.Er_BES, design.Pr_BES, design.DoD])[:, np.newaxis]*np.array([[1+Delta, 1-Delta],\
                                                                                        [1+Delta, 1-Delta],\
                                                                                        [1+Delta, 1-Delta]])
                                
    DF1 = np.zeros(X.shape)
    DF2 = np.zeros(X.shape)

    rows, columns = X.shape
    for i in range(rows): #there are 3 rows, Er, Pr, DoD
        for j in range(columns):

            x = X[i,j]

            if i == 0: #row 0 is the row of the Er
                design.Er_BES = x 
            elif i == 1:
                design.Pr_BES = x
            else:
                design.DoD = x

            data, data_time = dispatcher_dsctd.MyFun(design, False)

            design.cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)

            F1, _ , _ = LCOS_funct.MyFun(design, \
                                    capex = data['BES CAPEX [million euros]'].values[0]*1e6, \
                                    opex = data['BES OPEX [million euros]'].values[0]*1e6,\
                                    E_dch = sum(data_time['P_dch']),\
                                    res_val_bin = True
                                    )
            DF1[i,j] = (F1 - F1_0)/F1_0*100 #percentual change in F1
            
            F2 = data['Emissions Cost [million euros]'].values[0]
            DF2[i,j] = (F2 - F2_0)/F2_0

            design = design_0

    return F1_0, DF1, F2_0, DF2

DF1, DF2 = MyFun(design, 0.1)
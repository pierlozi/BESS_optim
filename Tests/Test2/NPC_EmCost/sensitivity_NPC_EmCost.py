#%%
import numpy as np
import pandas as pd
import altair as alt

import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_dsctd, microgrid_design
from Core import rain_deg_funct, LCOS_funct

import importlib
importlib.reload(rain_deg_funct)
importlib.reload(LCOS_funct)

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

df = pd.read_excel('test_NSGAII_NPC_EmCost_2.xlsx', index_col = 0, header= 0)

df_sorted = df.sort_values(by=['NPC'], ignore_index=True)
df_sorted['gamma'] = df.Er/df.Pr


def MyFun(design, Delta): #design has to have Er/Pr/DoD _0
    
    design_0 = design

    data, data_time = dispatcher_dsctd.MyFun(design_0, False)

    design.cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)

    design.DG_CAPEX = data['DG cost [million euros]']
    design.DG_OPEX = data['Fuel Cost [million euros]']

    _, _ , F1_0, _ = LCOS_funct.MyFun(design, E_dch = sum(data_time['P_dch']),\
                            res_val_bin = True
                            )

    F2_0 = data['Emissions Cost [million euros]'].values[0]

    dx = np.array([[1-Delta, 1+Delta],\
                    [1-Delta, 1+Delta],\
                    [1-Delta, 1+Delta]])

    X = np.array([design.Er_BES, design.Pr_BES, design.DoD])[:, np.newaxis]*dx
                             
    DF1 = np.zeros(X.shape)
    DF2 = np.zeros(X.shape)

    df_F1 = pd.DataFrame(columns = ['var', 'negative', 'positive'])
    df_F2 = pd.DataFrame(columns = ['var', 'negative', 'positive'])
    df_DEr = pd.DataFrame(columns = ['DX', 'DF1', 'DF2'])
    df_DPr = pd.DataFrame(columns = ['DX', 'DF1', 'DF2'])
    df_DDoD = pd.DataFrame(columns = ['DX', 'DF1', 'DF2'])
    
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

            design.DG_CAPEX = data['DG cost [million euros]']
            design.DG_OPEX = data['Fuel Cost [million euros]']
            
            _, _ , F1, _ = LCOS_funct.MyFun(design, E_dch = sum(data_time['P_dch']),\
                                    res_val_bin = True
                                    )
            DF1[i,j] = (F1_0[0] - F1[0])/F1_0[0] #normalized variation in F1    

            
            F2 = data['Emissions Cost [million euros]'].values[0]

            DF2[i,j] = (F2_0 - F2)/F2_0

            design = design_0
    
    df_F1['var'] = ['Er','Pr', 'DoD']
    df_F1.negative = DF1[:,0]
    df_F1.positive = DF1[:,1]

    df_DEr.DX = df_DPr.DX = df_DDoD.DX = [-Delta, Delta]

    df_DEr.DF1 = DF1[0]
    df_DEr.DF2 = DF2[0]

    df_DPr.DF1 = DF1[1]
    df_DPr.DF2 = DF2[1]

    df_DDoD.DF1 = DF1[2]
    df_DDoD.DF2 = DF2[2]


    df_F2['var'] = ['Er','Pr', 'DoD']
    df_F2.negative = DF2[:,0]
    df_F2.positive = DF2[:,1]

    
    return F1_0, F2_0, df_DEr, df_DPr, df_DDoD, DF1, DF2

#%%
idx = 21
design = microgrid_design.MG(Pr_BES=df_sorted.iloc[idx].Pr, \
                                Er_BES=df_sorted.iloc[idx].Er, \
                                DoD = df_sorted.iloc[idx].Er,\
                                P_load=P_load, \
                                P_ren=P_ren_read
                                )

F1_0, F2_0, df_DEr, df_DPr, df_DDoD, DF1, DF2 = MyFun(design, 0.1)

#%% Saving the sensitivity data
dictionary = {
    'DEr': df_DEr,
    'DPr': df_DPr,
    'DDoD': df_DDoD
}

# Iterate over each key-value pair in the dictionary
for key, df in dictionary.items():
    # Add the 'Key' column with the key value to the DataFrame
    df['Key'] = key
    
# Concatenate the DataFrames into a single DataFrame
df_combined = pd.concat(dictionary.values())

# Save the combined DataFrame to a CSV file
df_combined.to_csv('sensitivity_NPC_EmCost.csv', index=False)

#%% Reading the sensitivity data
# Read the CSV file into a DataFrame
df_combined = pd.read_csv('sensitivity_NPC_EmCost.csv')

# Create a dictionary to store the separate DataFrames
data = {}

# Iterate over unique keys in the 'Key' column
for key in df_combined['Key'].unique():
    # Filter the DataFrame based on the key
    df = df_combined[df_combined['Key'] == key].drop(columns='Key')
    
    # Store the DataFrame with the key as the variable name
    data[f"df_{key}"] = df

# Access the separate DataFrames using their variable names
df_DEr = data['df_DEr']
df_DPr = data['df_DPr']
df_DDoD = data['df_DDoD']

#%%
# xdom1 = max(df_F1.iloc[:, 1:].abs().max()*1.1)
# xdom2 = max(df_F2.iloc[:, 1:].abs().max()*1.1)
# opacity = 0.7

# chart_F1_1 = alt.Chart(
#                 df_F1,
#                 title = alt.Title(
#                 "Sensitivity LCOS",
#                 subtitle = "Variation of 10% in the variables"
#                 )).mark_bar(opacity=opacity,color = 'red').encode(
#             alt.X('positive').axis(format='%').title('[%]').scale(domain = (-xdom1,xdom1)),
#             alt.Y('var').axis().title('Variable')
#         )
# chart_F1_2 = alt.Chart(
#                 df_F1,
#                 title = alt.Title(
#                 "Sensitivity LCOS",
#                 subtitle = "Variation of 10% in the variables"
#                 )).mark_bar(opacity=opacity,color='blue').encode(
#             alt.X('negative').axis(format='%').title('[%]').scale(domain = (-xdom1,xdom1)),
#             alt.Y('var').axis().title('Variable')
#         )
# chart_F1 = chart_F1_1+chart_F1_2

# chart_F2_1 = alt.Chart(
#                 df_F2,
#                 title = alt.Title(
#                 "Sensitivity Emissions",
#                 subtitle = "Variation of 10% in the variables"
#                 )).mark_bar(opacity=opacity,color = 'red').encode(
#             alt.X('positive').axis(format='%').title('[%]').scale(domain = (-xdom2,xdom2)),
#             alt.Y('var').axis().title('Variable')
#         )
# chart_F2_2 = alt.Chart(
#                 df_F2,
#                 title = alt.Title(
#                 "Sensitivity Emissions",
#                 subtitle = "Variation of 10% in the variables"
#                 )).mark_bar(opacity=opacity,color='blue').encode(
#             alt.X('negative').axis(format='%').title('[%]').scale(domain = (-xdom2,xdom2)),
#             alt.Y('var').axis().title('Variable')
#         )
# chart_F2 = chart_F2_1+chart_F2_2
# chart_sensit = alt.vconcat(chart_F1, chart_F2)
# chart_sensit
# %%

DF1 = np.array([df_DEr['DF1'].abs().max(), df_DPr['DF1'].abs().max(), df_DDoD['DF1'].abs().max()])
DF2 = np.array([df_DEr['DF2'].abs().max(), df_DPr['DF2'].abs().max(), df_DDoD['DF2'].abs().max()])
xdom = np.maximum(DF1.max(), DF2.max())*1.1


width = 175
height = 200

# unicode_Delta = \u0394
#unicode_sub_0 = \u2080

titlef1 = '\u0394NPC/NPC\u2080'
titlef2 = '\u0394EC/EC\u2080'

titlex1 = '\u0394Er/Er\u2080'
titlex2 = '\u0394Pr/Pr\u2080'
titlex3 = '\u0394DoD/DoD\u2080'

chart1 = alt.Chart(df_DEr).mark_bar(size = 20).encode(
    alt.X('DX', axis = alt.Axis(values = list([-0.1, -0.05, 0, 0.05, 0.1]), gridWidth=0.5)).axis(format = '%').title(titlex1),
    alt.Y('DF1').axis(format = '%').scale(domain = (-xdom,xdom)).title(titlef1),
    color = alt.condition(
        alt.datum.DF1 > 0,
        alt.value("red"),
        alt.value("green")
    )
).properties(
            width = width,
            height = height
        )
chart2 = alt.Chart(df_DPr).mark_bar(size = 20).encode(
    alt.X('DX', axis = alt.Axis(values = list([-0.1, -0.05, 0, 0.05, 0.1]), gridWidth=0.5)).axis(format = '%').title(titlex2),
    alt.Y('DF1').axis(format = '%').scale(domain = (-xdom,xdom)).title(titlef1),
    color = alt.condition(
        alt.datum.DF1 > 0,
        alt.value("red"),
        alt.value("green")
    )
).properties(
            width = width,
            height = height
        )

chart3 = alt.Chart(df_DDoD).mark_bar(size = 20).encode(
    alt.X('DX', axis = alt.Axis(values = list([-0.1, -0.05, 0, 0.05, 0.1]), gridWidth=0.5)).axis(format = '%').title(titlex3),
    alt.Y('DF1').axis(format = '%').scale(domain = (-xdom,xdom)).title(titlef1),
    color = alt.condition(
        alt.datum.DF1 > 0,
        alt.value("red"),
        alt.value("green")
    )
).properties(
            width = width,
            height = height
        )

chart4 = alt.Chart(df_DEr).mark_bar(size = 20).encode(
    alt.X('DX', axis = alt.Axis(values = list([-0.1, -0.05, 0, 0.05, 0.1]), gridWidth=0.5)).axis(format = '%').title(titlex1),
    alt.Y('DF2').axis(format = '%').scale(domain = (-xdom,xdom)).title(titlef2),
    color = alt.condition(
        alt.datum.DF2 > 0,
        alt.value("red"),
        alt.value("green")
    )
).properties(
            width = width,
            height = height
        )
chart5 = alt.Chart(df_DPr).mark_bar(size = 20).encode(
    alt.X('DX', axis = alt.Axis(values = list([-0.1, -0.05, 0, 0.05, 0.1]), gridWidth=0.5)).axis(format = '%').title(titlex2),
    alt.Y('DF2').axis(format = '%').scale(domain = (-xdom,xdom)).title(titlef2),
    color = alt.condition(
        alt.datum.DF2 > 0,
        alt.value("red"),
        alt.value("green")
    )
).properties(
            width = width,
            height = height
        )

chart6 = alt.Chart(df_DDoD).mark_bar(size = 20).encode(
    alt.X('DX', axis = alt.Axis(values = list([-0.1, -0.05, 0, 0.05, 0.1]), gridWidth=0.5)).axis(format = '%').title(titlex3),
    alt.Y('DF2').axis(format = '%').scale(domain = (-xdom,xdom)).title(titlef2),
    color = alt.condition(
        alt.datum.DF2 > 0,
        alt.value("red"),
        alt.value("green")
    )
).properties(
            width = width,
            height = height
        )

#%%
spacing = 40

chartF1 = alt.hconcat(chart1, chart2, chart3, spacing = spacing)
chartF2 = alt.hconcat(chart4, chart5, chart6, spacing = spacing)

chart_sensit = alt.vconcat(chartF1, chartF2, spacing = spacing).configure_axis(
            labelFontSize = 15,
            titleFontSize = 20
        )

chart_sensit
#%%
chart_sensit.save('sensit_NPC_EmCost_2.png')

# %%

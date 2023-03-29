import pandas as pd
import numpy as np

df = pd.read_excel("test2_2_df_load_0_145.xlsx")

A = 3
B = 1
# P_prod(t) = A*1e6 + B*1e6*sin(2*np.pi*t/24)
# P_load(t) = (A+B)*C
# C = [(A-B)*(1-'Load [%]'/100)]/(A+B)

# Define a lambda function to perform the calculations for each row
calc_c = lambda row: ((A-B)*(1+row['Load [%]']/100))/(A+B)


# Apply the lambda function to each row of the 'Load [%]' column
df['C'] = df.apply(calc_c, axis=1)
df[['Load [%]','Pr_diesel [MW]', 'C']]
P_dg = (A+B)*df['C'] - A
df['P_dg_check'] = P_d
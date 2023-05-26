import altair as alt
from vega_datasets import data
import pandas as pd

# cars = data.cars()

# alt.Chart(cars).mark_point().encode(
#     x = 'Horsepower',
#     y = 'Miles_per_Gallon',
#     color='Origin'
# ).interactive()

results = pd.read_excel('Tests/Test6/res_GA_LCOS_rnflw_10pop_10gen.xlsx')
print(results.columns)

alt.Chart(results).mark_point().encode(
    x = 'LCOS [€/MWh]',
    y = 'Emissions Cost [million€]',
).interactive()
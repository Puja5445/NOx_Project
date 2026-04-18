import pandas as pd

df = pd.read_excel("Data/Furnace7_NOx_data.xlsx")

print(df.head())
print(df.shape)
print(df.columns)
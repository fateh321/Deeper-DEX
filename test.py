import pandas as pd
import numpy as np

df_temp = pd.read_csv('ALICEUSDT-1m-2023-01-07.csv',names=["Time","Alice","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = df_temp[df_temp.columns[1]]
df_temp = pd.read_csv('AUDIOUSDT-1m-2023-01-07.csv',names=["Time","Audio","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
df_temp = pd.read_csv('ATOMUSDT-1m-2023-01-07.csv',names=["Time","Atom","Low","High","Close","Junk1","Junk2","Junk3","Junk4","Junk5","Junk6","Junk7"])
df = pd.concat([df, df_temp[df_temp.columns[1]]], axis=1)
#df.columns =['Price']
dfArr = df.to_numpy()
print(df.iloc[0].to_numpy())
print(dfArr[100][1])
print(df.max().to_numpy())
print(df[df.columns[1]].max()) 
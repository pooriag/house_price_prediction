import pandas as pnd
import numpy as np
import neural_network

df = pnd.read_csv('Housing.csv')


df = df.replace(to_replace='yes', value=1)
df = df.replace(to_replace='no', value=0)
df = df.replace(to_replace='furnished', value=2)
df = df.replace(to_replace='semi-furnished', value=1)
df = df.replace(to_replace='unfurnished', value=0)
#print(df['mainroad'])
list = df.iloc[[0, 1, 2]].values.tolist()
#print(list)
nn = neural_network.house_price_network(13)
nn.predic(list)
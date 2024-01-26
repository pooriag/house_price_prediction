import random
import pandas as pnd
import numpy as np
import neural_network
import matplotlib.pyplot as plt

def data_clean(df):
    df = df.replace(to_replace='yes', value=1)
    df = df.replace(to_replace='no', value=0)
    df = df.replace(to_replace='furnished', value=2)
    df = df.replace(to_replace='semi-furnished', value=1)
    df = df.replace(to_replace='unfurnished', value=0)
    #print(df.isnull().any(axis=1))
    return df

def normalizing(df):
    mean = df[df.columns[1:]].mean()
    std = df[df.columns[1:]].std()
    df[df.columns[1:]] = (df[df.columns[1:]] - mean) / std
    return df

def random_sample(df, start, end, number):
    return df.iloc[[random.randint(start, end) for i in range(number)]]

def train(df, number, iteration):

    for i in range(iteration):
        sample = random_sample(df, 0, len(df) - 1, number)
        nn.train(sample[sample.columns[1:]].values.tolist(), sample[sample.columns[0]].values.tolist())

def plot_losses(losses):
    plt.figure()

    for i in range(0, len(losses), 100):
        plt.plot(i, losses[i], 'ro')
    plt.show()


df = pnd.read_csv('Housing.csv')
df = data_clean(df)
normalized_df = normalizing(df)

df_train = normalized_df[normalized_df.index % 2 == 0]
df_test = normalized_df[(normalized_df.index % 2 == 1) & (normalized_df.index % 3 != 0)]
df_eval = normalized_df[(normalized_df.index % 2 == 1) & (normalized_df.index % 3 == 0)]

nn = neural_network.house_price_network(12)

train(df_train, 50,  40)

test_sample = random_sample(df_test, 0, len(df_test) - 1, 15)
for i in range(len(test_sample)):
    print(f'actuale value: {test_sample["price"].iloc[i]}')
    print(f'predicted value: {nn.predict(test_sample.iloc[i][1:].values.tolist())}')

plot_losses(nn.losses)
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

def random_train_sample(df, start, end, number):
    return df.iloc[[random.randint(start, end) for i in range(number)]]

def train(df, start, end, number, iteration):

    for i in range(iteration):
        sample = random_train_sample(df, start, end, number)
        nn.train(sample[sample.columns[1:]].values.tolist(), sample[sample.columns[0]].values.tolist())

def plot_losses(losses):
    plt.figure()

    for i in range(0, len(losses), 100):
        plt.plot(i, losses[i], 'ro')
    plt.show()


df = pnd.read_csv('Housing.csv')
df = data_clean(df)
normalized_df = normalizing(df)
nn = neural_network.house_price_network(12)
train(normalized_df, 1, 200, 30,  40)
for i in range(205, 215):
    print(f'actuale value{df["price"].iloc[i]}')
    print(f'predicted value{nn.predict(normalized_df.iloc[i][1:].values.tolist())}')

plot_losses(nn.losses)
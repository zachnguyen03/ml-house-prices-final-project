import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass


def plot_bar(x, y, xlabel, ylabel, title, plot_size):
    f, ax = plt.subplots(figsize=plot_size)
    plt.xticks(rotation='90')
    sns.barplot(x=x, y=y)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=15)
    
def plot_heatmap(x, plot_size):
    plt.subplots(figsize=plot_size)
    sns.heatmap(x, vmax=0.9, square=True)
    
def plot_dist(x, fit):
    sns.distplot(x, fit=fit)
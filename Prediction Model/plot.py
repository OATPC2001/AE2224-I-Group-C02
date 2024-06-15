import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator


def plot_histograms(X, Y, bins, filename):
    
    X = pd.to_datetime(X,format='%d-%m-%Y')
    min_date = X.min()
    max_date = X.max()
    num_bins = int((max_date - min_date).days / bins) + 1


    plt.figure(figsize=(10, 6))
    plt.hist(X, bins=num_bins, weights=Y, edgecolor='black')  # Using weights to account for flight numbers
    plt.xlabel('Date')
    plt.ylabel('Flight Number')
    plt.title('Flight Number Histogram by Date')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    plt.savefig(filename)

def plot_scatter(X, y, filename,color):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.grid(True)
    ax.scatter(X, y, color=color)
    ax.set_xlabel('Years')
    ax.set_ylabel('Emissions in kg')
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    #ax.set_yscale('log')
    plt.savefig(filename)


def plot_line(X, y,filename,color):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.grid(True)
    ax.plot(X, y/1e6, color=color, label='CO2', linewidth=2)
    ax.set_xlabel('Years')
    ax.set_ylabel('Emissions (in thousands of tonnes)')
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    plt.legend()


    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    minor_locator = AutoMinorLocator(4)  # Three minor ticks between major ticks
    ax.xaxis.set_minor_locator(minor_locator)

    #ax.set_yscale('log')
    plt.savefig(filename)


def plot_dual_scatter(X, Y, X2, y2, filename):
    plt.scatter(X, Y, color='blue', label='X-Y')
    plt.scatter(X2, y2, color='red', label='X_test-y_pred')

    # Customize plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Save plot
    plt.savefig(filename)
    plt.show()



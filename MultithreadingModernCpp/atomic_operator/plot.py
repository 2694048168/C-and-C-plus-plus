# Python script to plot outputs of Google Benchmark in CSV format
# By Ari Saif
# ------------------------------------------------------------------------------
# usage: plot.py [-h] [--xLabel [XLABEL]] [--title [TITLE]]
#                [--series SERIES [SERIES ...]]
#                INPUT [INPUT ...]

# Plots benchmark CSV outputs.

# positional arguments:
#   INPUT                 input file name

# optional arguments:
#   -h, --help            show this help message and exit
#   --xLabel [XLABEL]     Label for the X axis
#   --title [TITLE]       Title of the plot
#   --series SERIES [SERIES ...]
#                         Series names to be shown on the plot
# ------------------------------------------------------------------------------
# Example call:
# python3 plot.py src/benchmark/outputs/step4_random_with_check.csv --series "real_time_BM_IntroSortPar" "real_time_BM_StdSort"
# ------------------------------------------------------------------------------
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import reduce

# Set input arguments
parser = argparse.ArgumentParser(description='Plots benchmark CSV outputs.')
parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                    help='input file name')

parser.add_argument('--xLabel', dest='xLabel',
                    default="n", type=str, nargs='?',
                    help='Label for the X axis')

parser.add_argument('--title', dest='title',
                    default='Run Time', type=str, nargs='?',
                    help='Title of the plot')

parser.add_argument('--series', dest='series',
                    type=str, nargs='+',
                    help='Series names to be shown on the plot')

args = parser.parse_args()

# Read input CSV file
df = pd.read_csv(args.input[0])

#Remove '/' from the names and create a new column representing the size (n)
df[['name', 'n']] = df.name.str.split("/", expand=True)
df = df.filter(['name', 'n', 'real_time', 'cpu_time'])

# Filter out BigO and RMS rows
df = df[~df['name'].str.contains("BigO")]
df = df[~df['name'].str.contains("RMS")]

#Only keep n, name, and real_time columns
df = df.filter(['n', 'name', 'real_time'])
print(df)

# Unstack data from various benchmarks and put them in new columns
dfs = [
    g.drop('name', 1).add_suffix(f'_{k}').rename({f'n_{k}': 'n'}, axis=1)
    for k, g in df.groupby('name')
]

df1 = reduce(lambda x, y: pd.merge(x, y, on='n'), dfs)


# Filter out the series that we want to be in the plot using --series
if args.series and len(args.series) > 0:
    columns_to_show = ['n'] + args.series
    df1 = df1.filter(columns_to_show)

print(df1)

#Remove real_time_BM_ prefix from column names
df1.columns = [x.strip().replace('real_time_BM_', '') for x in df1.columns]

#Draw the plot
ax = df1.plot(x='n', title=args.title,
              figsize=(15, 10), legend=True, fontsize=12, rot=0)

ax.set_xticks(np.arange(len(df1['n'])))
ax.set_xticklabels(df1['n'], rotation=45)
ax.legend(fontsize=16)

plt.show()

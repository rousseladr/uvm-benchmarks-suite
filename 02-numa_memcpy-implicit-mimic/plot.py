# doc : https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# doc : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import sys

if len(sys.argv) <= 1:
    print("No arguments given")
    print("Usage:\n\t python3 plot.py <SIZE-IN-MB>")
    sys.exit(0)

f=sys.argv[1]+"-MB_numa_implicit-mimic_gbs.csv"

df = pd.read_csv(f, delimiter='\t')

y = df["gpu"]
x = df["core"]
A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["HostToDevice"]

print(A.min())
print(A.max())

sns.heatmap(A, vmin=105, vmax=137, center=121, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Async. Explicit Transfers")
plt.suptitle("Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=sys.argv[1]+"-MB_A100_HtD_async-explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["DeviceToHost"]

print(A.min())
print(A.max())

sns.heatmap(A, vmin=51, vmax=119, center=85, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle("Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=sys.argv[1]+"-MB_A100_HtD_async-explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

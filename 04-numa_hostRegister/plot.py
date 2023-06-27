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

f=sys.argv[1]+"-MB_numa_hostRegister_gbs.csv"

df = pd.read_csv(f, delimiter='\t')

y = df["gpu"]
x = df["core"]
A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["HostToDevice"]

B = np.zeros((x.max()+1,y.max()+1))
B[x,y] = df["DeviceToHost"]

min2=A.min()
if B.min() <= min2:
    min2 = B.min()
min2 = min2-1

max2=A.max()
if B.max() >= max2:
    max2 = B.max()
max2 = max2+1

med = (max2 + min2) / 2

print(min2)
print(max2)
print(med)

sns.heatmap(A, vmin=min2, vmax=max2, center=med, cmap="ocean")
plt.title("Throughput (GB/s) of cudaHostRegister on A100 - Explicit")
plt.suptitle(sys.argv[1]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=sys.argv[1]+"-MB_A100_HtD_hostregister.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B, vmin=min2, vmax=max2, center=med, cmap="ocean")
plt.title("Throughput (GB/s) of cudaHostRegister on A100 - Explicit")
plt.suptitle(sys.argv[1]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=sys.argv[1]+"-MB_A100_DtH_hostregister.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

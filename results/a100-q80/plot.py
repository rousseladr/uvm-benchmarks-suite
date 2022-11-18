# doc : https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# doc : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('a100-q80_explicit.csv', delimiter='\t')

y = df["gpu"]
x = df["core"]
A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["HostToDevice"]

print(A.min())
print(A.max())

sns.heatmap(A, vmin=105, vmax=175, center=140, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100-q80 - Explicit")
plt.suptitle("Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
plt.savefig("A100-q80_HtD_explicit.pdf", format="pdf", bbox_inches="tight")

plt.clf()

A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["DeviceToHost"]

print(A.min())
print(A.max())

sns.heatmap(A, vmin=72, vmax=172, center=124, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100-q80 - Explicit")
plt.suptitle("Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
plt.savefig("A100-q80_DtH_explicit.pdf", format="pdf", bbox_inches="tight")

plt.clf()

df = pd.read_csv('a100-q80_implicit.csv', delimiter='\t')

y = df["gpu"]
x = df["core"]
A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["HostToDevice"]

print(A.min())
print(A.max())

sns.heatmap(A, vmin=105, vmax=145, center=140, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100-q80 - Implicit (Managed)")
plt.suptitle("Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
plt.savefig("A100-q80_HtD_implicit.pdf", format="pdf", bbox_inches="tight")

plt.clf()

A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["DeviceToHost"]

print(A.min())
print(A.max())

sns.heatmap(A, vmin=72, vmax=172, center=124, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100-q80 - Implicit (Managed)")
plt.suptitle("Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
plt.savefig("A100-q80_DtH_implicit.pdf", format="pdf", bbox_inches="tight")

plt.clf()

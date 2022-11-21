# doc : https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# doc : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('numa_explicit_gbs.csv', delimiter='\t')

y = df["gpu"]
x = df["core"]
A = np.zeros((x.max()+1,y.max()+1))
A[x,y] = df["HostToDevice"]

print(A.min())
print(A.max())

sns.heatmap(A, vmin=105, vmax=137, center=121, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle("Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
plt.savefig("A100_HtD_explicit.pdf", format="pdf", bbox_inches="tight")

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
plt.savefig("A100_DtH_explicit.pdf", format="pdf", bbox_inches="tight")

plt.clf()

#  df = pd.read_csv('numa_implicit_gbs.csv', delimiter='\t')
#
#  y = df["gpu"]
#  x = df["core"]
#  A = np.zeros((x.max()+1,y.max()+1))
#  A[x,y] = df["HostToDevice"]
#
#  print(A.min())
#  print(A.max())
#
#  sns.heatmap(A, vmin=105, vmax=137, center=121, cmap="ocean")
#  plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Implicit (Managed)")
#  plt.suptitle("Host To Device")
#  plt.xlabel('GPU Number')
#  plt.ylabel('Core Number')
#  plt.savefig("A100_HtD_implicit.pdf", format="pdf", bbox_inches="tight")
#
#  plt.clf()
#
#  A = np.zeros((x.max()+1,y.max()+1))
#  A[x,y] = df["DeviceToHost"]
#
#  print(A.min())
#  print(A.max())
#
#  sns.heatmap(A, vmin=51, vmax=119, center=85, cmap="ocean")
#  plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Implicit (Managed)")
#  plt.suptitle("Device To Host")
#  plt.xlabel('GPU Number')
#  plt.ylabel('Core Number')
#  plt.savefig("A100_DtH_implicit.pdf", format="pdf", bbox_inches="tight")
#
#  plt.clf()

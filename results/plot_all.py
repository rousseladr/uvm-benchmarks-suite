# doc : https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# doc : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import sys

if len(sys.argv) <= 2:
    print("No arguments given")
    print("Usage:\n\t python3 plot.py <SYSTEM ID> <SIZE-IN-MB>")
    sys.exit(0)

f0=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_implicit_gbs.csv"
f1=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_explicit_gbs.csv"
f2=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_implicit-mimic_gbs.csv"
f3=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_memcpyasync_gbs.csv"

df0 = pd.read_csv(f0, delimiter='\t')
df1 = pd.read_csv(f1, delimiter='\t')
df2 = pd.read_csv(f2, delimiter='\t')
df3 = pd.read_csv(f3, delimiter='\t')

y0 = df0["gpu"]
x0 = df0["core"]
A0 = np.zeros((x0.max()+1,y0.max()+1))
A0[x0,y0] = df0["HostToDevice"]
B0 = np.zeros((x0.max()+1,y0.max()+1))
B0[x0,y0] = df0["DeviceToHost"]

y1 = df1["gpu"]
x1 = df1["core"]
A1 = np.zeros((x1.max()+1,y1.max()+1))
A1[x1,y1] = df1["HostToDevice"]
B1 = np.zeros((x1.max()+1,y1.max()+1))
B1[x1,y1] = df1["DeviceToHost"]

y2 = df2["gpu"]
x2 = df2["core"]
A2 = np.zeros((x2.max()+1,y2.max()+1))
A2[x2,y2] = df2["HostToDevice"]
B2 = np.zeros((x2.max()+1,y2.max()+1))
B2[x2,y2] = df2["DeviceToHost"]

y3 = df3["gpu"]
x3 = df3["core"]
A3 = np.zeros((x3.max()+1,y3.max()+1))
A3[x3,y3] = df3["HostToDevice"]
B3 = np.zeros((x3.max()+1,y3.max()+1))
B3[x3,y3] = df3["DeviceToHost"]

minallA = np.array([A0.min(), A1.min(), A2.min(), A3.min()])
maxallA = np.array([A0.max(), A1.max(), A2.max(), A3.max()])

minallB = np.array([B0.min(), B1.min(), B2.min(), B3.min()])
maxallB = np.array([B0.max(), B1.max(), B2.max(), B3.max()])

minag = minallA.min()
maxag = maxallA.max()
meda = (maxag+minag)/2

minbg = minallB.min()
maxbg = maxallB.max()
medb = (maxbg+minbg)/2

print(maxallA.max())
print(minallA.min())
print(meda)

print(minallB.min())
print(maxallB.max())
print(medb)

res_dir=sys.argv[1]+"/"+sys.argv[2]+"MB/pdf/"

sns.heatmap(A0, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_HtD_explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B0, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_DtH_explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(A1, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_HtD_managed.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B1, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_DtH_managed.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(A2, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_HtD_implicit-mimic.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B3, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_DtH_implicit-mimic.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(A3, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_HtD_async-explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B3, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of cudaMemcpy on A100 - Explicit")
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_A100_DtH_async-explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

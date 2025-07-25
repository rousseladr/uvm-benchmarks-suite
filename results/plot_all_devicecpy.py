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

f0=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_explicit_gbs.csv"
f1=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_explicit_gbs_copy_host.csv"
f2=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_explicit_gbs_copy_device_hip.csv"
#f3=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_explicit_gbs_copy_device_malloc.csv"
f4=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_memcpyasync_gbs.csv"
f5=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_memcpyasync_gbs_copy_host.csv"
f6=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_memcpyasync_gbs_copy_device_hip.csv"
#f7=sys.argv[1]+"/"+sys.argv[2]+"MB/csv/"+sys.argv[2]+"-MB_numa_memcpyasync_gbs_copy_device_malloc.csv"

df0 = pd.read_csv(f0, delimiter='\t')
df1 = pd.read_csv(f1, delimiter='\t')
df2 = pd.read_csv(f2, delimiter='\t')
#df3 = pd.read_csv(f3, delimiter='\t')
df4 = pd.read_csv(f4, delimiter='\t')
df5 = pd.read_csv(f5, delimiter='\t')
df6 = pd.read_csv(f6, delimiter='\t')
#df7 = pd.read_csv(f7, delimiter='\t')

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

#y3 = df3["gpu"]
#x3 = df3["core"]
#A3 = np.zeros((x3.max()+1,y3.max()+1))
#A3[x3,y3] = df3["HostToDevice"]
#B3 = np.zeros((x3.max()+1,y3.max()+1))
#B3[x3,y3] = df3["DeviceToHost"]

y4 = df4["gpu"]
x4 = df4["core"]
A4 = np.zeros((x4.max()+1,y4.max()+1))
A4[x4,y4] = df4["HostToDevice"]
B4 = np.zeros((x4.max()+1,y4.max()+1))
B4[x4,y4] = df4["DeviceToHost"]

y5 = df5["gpu"]
x5 = df5["core"]
A5 = np.zeros((x5.max()+1,y5.max()+1))
A5[x5,y5] = df5["HostToDevice"]
B5 = np.zeros((x5.max()+1,y5.max()+1))
B5[x5,y5] = df5["DeviceToHost"]

y6 = df6["gpu"]
x6 = df6["core"]
A6 = np.zeros((x6.max()+1,y6.max()+1))
A6[x6,y6] = df6["HostToDevice"]
B6 = np.zeros((x6.max()+1,y6.max()+1))
B6[x6,y6] = df6["DeviceToHost"]

#y7 = df7["gpu"]
#x7 = df7["core"]
#A7 = np.zeros((x7.max()+1,y7.max()+1))
#A7[x7,y7] = df7["HostToDevice"]
#B7 = np.zeros((x7.max()+1,y7.max()+1))
#B7[x7,y7] = df7["DeviceToHost"]

#minallA = np.array([A0.min(), A1.min(), A2.min(), A3.min(), A4.min(), A5.min(), A6.min(), A7.min()])
#maxallA = np.array([A0.max(), A1.max(), A2.max(), A3.max(), A4.max(), A5.max(), A6.max(), A7.max()])
#
#minallB = np.array([B0.min(), B1.min(), B2.min(), B3.min(), B4.min(), B5.min(), B6.min(), B7.min()])
#maxallB = np.array([B0.max(), B1.max(), B2.max(), B3.max(), B4.max(), B5.max(), B6.max(), B7.max()])

minallA = np.array([A0.min(), A1.min(), A2.min(), A4.min(), A5.min(), A6.min()])
maxallA = np.array([A0.max(), A1.max(), A2.max(), A4.max(), A5.max(), A6.max()])

minallB = np.array([B0.min(), B1.min(), B2.min(), B4.min(), B5.min(), B6.min()])
maxallB = np.array([B0.max(), B1.max(), B2.max(), B4.max(), B5.max(), B6.max()])

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
plt.title("Throughput (GB/s) of hipMemcpy on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_HtD_explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B0, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of hipMemcpy on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_DtH_explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(A1, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of copies from host (malloc allocation)  on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_HtD_copy_host.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B1, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of copies from host (malloc allocation)  on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_DtH_copy_host.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(A2, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of copies from device (hipMalloc allocation) on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_HtD_copy_device_hip.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B2, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of copies from device (hipMalloc allocation) on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_DtH_copy_device_hip.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

#sns.heatmap(A3, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
#plt.title("Throughput (GB/s) of copies from device (malloc allocation) on "+sys.argv[1])
#plt.suptitle(sys.argv[2]+"MB - Host To Device")
#plt.xlabel('GPU Number')
#plt.ylabel('Core Number')
#fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_HtD_copy_device_malloc.pdf"
#plt.savefig(fsave, format="pdf", bbox_inches="tight")
#
#plt.clf()
#
#sns.heatmap(B3, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
#plt.title("Throughput (GB/s) of copies from device (malloc allocation) on "+sys.argv[1])
#plt.suptitle(sys.argv[2]+"MB - Device To Host")
#plt.xlabel('GPU Number')
#plt.ylabel('Core Number')
#fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_DtH_copy_device_malloc.pdf"
#plt.savefig(fsave, format="pdf", bbox_inches="tight")
#
#plt.clf()

sns.heatmap(A4, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of hipMemcpyAsync on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_async_HtD_explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B4, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of hipMemcpyAsync on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_async_DtH_explicit.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(A5, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of copies from host (malloc allocation)  on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_async_HtD_copy_host.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B5, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of copies from host (malloc allocation)  on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_async_DtH_copy_host.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(A6, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
plt.title("Throughput (GB/s) of copies from device (hipMalloc allocation) on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Host To Device")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_HtD_async_copy_device_hip.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

sns.heatmap(B6, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
plt.title("Throughput (GB/s) of copies from device (hipMalloc allocation) on "+sys.argv[1])
plt.suptitle(sys.argv[2]+"MB - Device To Host")
plt.xlabel('GPU Number')
plt.ylabel('Core Number')
fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_async_DtH_copy_device_hip.pdf"
plt.savefig(fsave, format="pdf", bbox_inches="tight")

plt.clf()

#sns.heatmap(A7, vmin=minag, vmax=maxag, center=meda, cmap="ocean")
#plt.title("Throughput (GB/s) of copies from device (malloc allocation) on "+sys.argv[1])
#plt.suptitle(sys.argv[2]+"MB - Host To Device")
#plt.xlabel('GPU Number')
#plt.ylabel('Core Number')
#fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_async_HtD_copy_device_malloc.pdf"
#plt.savefig(fsave, format="pdf", bbox_inches="tight")
#
#plt.clf()
#
#sns.heatmap(B7, vmin=minbg, vmax=maxbg, center=medb, cmap="ocean")
#plt.title("Throughput (GB/s) of copies from device (malloc allocation) on "+sys.argv[1])
#plt.suptitle(sys.argv[2]+"MB - Device To Host")
#plt.xlabel('GPU Number')
#plt.ylabel('Core Number')
#fsave=res_dir+sys.argv[2]+"-MB_"+sys.argv[1]+"_async_DtH_copy_device_malloc.pdf"
#plt.savefig(fsave, format="pdf", bbox_inches="tight")
#
#plt.clf()

import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import colorsys

# Reading Args
name = "peppersc"
clusters = 2
if len(sys.argv)==2:
    if(sys.argv[1].isdigit()):
        clusters = int(sys.argv[1])
    else:
        name = sys.argv[1]
elif len(sys.argv)==3:
    if(sys.argv[1].isdigit() and not sys.argv[2].isdigit()):
        clusters = int(sys.argv[1])
        name = sys.argv[2]
    elif(sys.argv[2].isdigit() and not sys.argv[1].isdigit()):
        clusters = int(sys.argv[2])
        name = sys.argv[1]
    else:
        print("Error: Incorrect arguments")
        sys.exit(-1)
else:
    print("Error: Too many arguments")
    sys.exit(-1)

print(("Using %d clusters"%clusters)+" for image "+name)

show=False
save=True

# Load data

df = pd.read_csv("../data/"+name+".csv")

max_x=df["x"].max()
max_y=df["y"].max()

# RGB color scheme
print("\nCalculating RGB Hierarchical Agglomerative Clustering")
hac = AgglomerativeClustering(n_clusters = clusters).fit(df[["r","g","b"]])
print("Done")

print("\nPostprocessing RGB pixels")
# calculating centers
centers = [[0,0,0] for i in range(clusters)]
n_label = [0 for i in range(clusters)]
df['label']=hac.labels_
for it,row in df.iterrows():
    n_label[row['label']]+=1
    centers = np.add(centers,[df['r'],df['g'],df['b']])

for c, n in zip(centers, n_label):
    c[0] = c[0]/n
    c[1] = c[1]/n
    c[2] = c[2]/n

img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
colors=[]
for it,row in df.iterrows():
    x,y,lab = int(row['x']),int(row['y']),int(row['label'])
    img_colored[x][y] = [int(round(c)) for c in centers[lab]]
    colors.append([c/255.0 for c in centers[lab]])
print("Done")

print("\nGenerating RGB Plots and images")
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['r'],df['g'],df['b'],c=colors)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_rgb_plot_hac.png"%clusters)
plt.cla()
plt.clf()

plt.imshow(img_colored)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_rgb_colored_hac.png"%clusters)
plt.cla()
plt.clf()
print("Done")

# HSV color scheme

# Mapping the colors
print("\nMapping HS(V) colors in 2D space")
data_hs = []
for it,row in df.iterrows():
    x,y,h,s = int(row['x']),int(row['y']),row['h'],row['s']
    hsx = s * np.cos(h*(np.pi*2))
    hsy = s * np.sin(h*(np.pi*2))
    data_hs.append([x,y,hsx,hsy])
df_hs = pd.DataFrame(data=data_hs, columns=["x","y","hsx","hsy"])
print("Done")

# plt.scatter(df_hs['hsx'],df_hs['hsy'])
# plt.show()

print("\nCalculating HS(V) Hierarchical Agglomerative Clustering")
hac = AgglomerativeClustering(n_clusters = clusters).fit(df_hs[["hsx","hsy"]])
print("Done")

# calculating centers
centers = [[0,0] for i in range(clusters)]
n_label = [0 for i in range(clusters)]
df_hs['label']=hac.labels_
for it,row in df.iterrows():
    n_label[row['label']]+=1
    centers = np.add(centers,[df_hs['hsx'],df['hsy']])

for c, n in zip(centers, n_label):
    c[0] = c[0]/n
    c[1] = c[1]/n

print("\nPostprocessing HS(V) pixels")
centers_color = []
for [hsx,hsy] in centers:
    h = np.arctan2(hsy,hsx)
    if h<0:
        h=(2*np.pi+h)
    h = h/(2*np.pi)
    s = np.sqrt(hsx**2+hsy**2)
    (r,g,b) = colorsys.hsv_to_rgb(h,s,200)
    centers_color.append([r,g,b])

img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
colors_hs = []
for it,row in df_hs.iterrows():
    x,y,lab = int(row['x']),int(row['y']),int(row['label'])
    img_colored[x][y] = [int(round(c)) for c in centers_color[lab]]
    colors_hs.append([c/255.0 for c in centers_color[lab]])
print("Done")

print("\nGenerating HS(V) Plots and images")
plt.cla()
plt.scatter(df_hs['hsx'],df_hs['hsy'],c=colors_hs)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_hsv_plot_hac.png"%clusters)
plt.cla()

plt.imshow(img_colored)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_hsv_colored_hac.png"%clusters)
plt.cla()
print("Done")

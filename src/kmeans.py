import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import colorsys

if len(sys.argv)==1:
    print("Using 2 clusters")
    clusters = 2
elif len(sys.argv)==2:
    clusters = int(sys.argv[1])
    print("Using %d clusters"%clusters)
else:
    print("Error: Too many arguments using k=2")

show=False
save=True
# load
name = "lena"
df = pd.read_csv("../data/"+name+".csv")

max_x=df["x"].max()
max_y=df["y"].max()

# RGB color scheme
print("\nCalculating RGB kmeans")
kmeans = KMeans(n_clusters = clusters).fit(df[["r","g","b"]])
print("Done")

#print(kmeans.cluster_centers_)
centers = kmeans.cluster_centers_

print("\nPostprocessing RGB pixels")
img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
colors=[]
for it,row in df.iterrows():
    x,y,r,g,b = int(row['x']),int(row['y']),row['r'],row['g'],row['b']
    [center] = kmeans.predict([[r,g,b]])
    img_colored[x][y] = [int(round(c)) for c in centers[center]]
    colors.append([c/255.0 for c in centers[center]])
print("Done")

print("\nGenerating RGB Plots and images")
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['r'],df['g'],df['b'],c=colors)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_rgb_plot.png"%clusters)
plt.cla()
plt.clf()

plt.imshow(img_colored)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_rgb_colored.png"%clusters)
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

print("\nCalculating HS(V) kmeans")
kmeans = KMeans(n_clusters = clusters).fit(df_hs[["hsx","hsy"]])
print("Done")

centers = kmeans.cluster_centers_

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
    x,y,hsx,hsy = int(row['x']),int(row['y']),row['hsx'],row['hsy']
    [center] = kmeans.predict([[hsx,hsy]])
    img_colored[x][y] = [int(round(c)) for c in centers_color[center]]
    colors_hs.append([c/255.0 for c in centers_color[center]])
print("Done")

print("\nGenerating HS(V) Plots and images")
plt.cla()
plt.scatter(df_hs['hsx'],df_hs['hsy'],c=colors_hs)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_hsv_plot.png"%clusters)
plt.cla()

plt.imshow(img_colored)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_hsv_colored.png"%clusters)
plt.cla()
print("Done")

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import colorsys

# load
name = "peppersc"
df = pd.read_csv("../data/"+name+".csv")

max_x=df["x"].max()
max_y=df["y"].max()

# RGB color scheme

kmeans = KMeans(n_clusters = 2).fit(df[["r","g","b"]])

#print(kmeans.cluster_centers_)
centers = kmeans.cluster_centers_

img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]

for it,row in df.iterrows():
    x,y,r,g,b = int(row['x']),int(row['y']),row['r'],row['g'],row['b']
    [center] = kmeans.predict([[r,g,b]])
    img_colored[x][y] = [int(round(c)) for c in centers[center]]

plt.imshow(img_colored)
plt.show()

# HSV color scheme
# Mapping the colors
data_hs = []
for it,row in df.iterrows():
    x,y,h,s = int(row['x']),int(row['y']),row['h'],row['s']
    hsx = s * np.cos(h*(np.pi*2))
    hsy = s * np.sin(h*(np.pi*2))
    data_hs.append([x,y,hsx,hsy])
df_hs = pd.DataFrame(data=data_hs, columns=["x","y","hsx","hsy"])

plt.scatter(df_hs['hsx'],df_hs['hsy'])
plt.show()

kmeans = KMeans(n_clusters = 2).fit(df_hs[["hsx","hsy"]])
centers = kmeans.cluster_centers_

centers_color = []
for [hsx,hsy] in centers:
    h = np.arctan2(hsy,hsx)
    if h<0:
        h=(2*np.pi+h)
    h = h/(2*np.pi)
    s = np.sqrt(hsx**2+hsy**2)
    (r,g,b) = colorsys.hsv_to_rgb(h,s,255)
    centers_color.append([r,g,b])

img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
colors = []
for it,row in df_hs.iterrows():
    x,y,hsx,hsy = int(row['x']),int(row['y']),row['hsx'],row['hsy']
    [center] = kmeans.predict([[hsx,hsy]])
    img_colored[x][y] = [int(round(c)) for c in centers_color[center]]
    colors.append([c/255.0 for c in centers_color[center]])

plt.scatter(df_hs['hsx'],df_hs['hsy'],c=colors)
plt.show()

plt.imshow(img_colored)
plt.show()

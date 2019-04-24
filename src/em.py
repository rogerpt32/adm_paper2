import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import colorsys

# Reading Args
name = "peppersc"
components = 2
if len(sys.argv)==2:
    if(sys.argv[1].isdigit()):
        components = int(sys.argv[1])
    else:
        name = sys.argv[1]
elif len(sys.argv)==3:
    if(sys.argv[1].isdigit() and not sys.argv[2].isdigit()):
        components = int(sys.argv[1])
        name = sys.argv[2]
    elif(sys.argv[2].isdigit() and not sys.argv[1].isdigit()):
        components = int(sys.argv[2])
        name = sys.argv[1]
    else:
        print("Error: Incorrect arguments")
elif len(sys.argv)>3:
    print("Error: Too many arguments")
    sys.exit(-1)

print(("Using %d components"%components)+" for image "+name)

show=False
save=True

# Load data

df = pd.read_csv("../data/"+name+".csv")

max_x=df["x"].max()
max_y=df["y"].max()

# RGB color scheme
print("\nCalculating RGB gmm")
gmm = GaussianMixture(n_components = components).fit(df[["r","g","b"]])
df['label'] = gmm.predict(df[['r','g','b']])
print("Done")

#print(gmm.means_)
means = gmm.means_

print("\nPostprocessing RGB pixels")
img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
colors=[]
for it,row in df.iterrows():
    x,y,lab = int(row['x']),int(row['y']),int(row['label'])
    img_colored[x][y] = [int(round(c)) for c in means[lab]]
    colors.append([c/255.0 for c in means[lab]])
print("Done")

print("\nGenerating RGB Plots and images")
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['r'],df['g'],df['b'],c=colors)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_rgb_plot_gmm.png"%components)
plt.cla()
plt.clf()

plt.imshow(img_colored)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_rgb_colored_gmm.png"%components)
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

print("\nCalculating HS(V) gmm")
gmm = GaussianMixture(n_components = components).fit(df_hs[["hsx","hsy"]])
df_hs['label'] = gmm.predict(df_hs[["hsx","hsy"]])
print("Done")

means = gmm.means_

print("\nPostprocessing HS(V) pixels")
means_color = []
for [hsx,hsy] in means:
    h = np.arctan2(hsy,hsx)
    if h<0:
        h=(2*np.pi+h)
    h = h/(2*np.pi)
    s = np.sqrt(hsx**2+hsy**2)
    (r,g,b) = colorsys.hsv_to_rgb(h,s,200)
    means_color.append([r,g,b])

img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
colors_hs = []
for it,row in df_hs.iterrows():
    x,y,lab = int(row['x']),int(row['y']),int(row['label'])
    img_colored[x][y] = [int(round(c)) for c in means_color[lab]]
    colors_hs.append([c/255.0 for c in means_color[lab]])
print("Done")

print("\nGenerating HS(V) Plots and images")
plt.cla()
plt.scatter(df_hs['hsx'],df_hs['hsy'],c=colors_hs)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_hsv_plot_gmm.png"%components)
plt.cla()

plt.imshow(img_colored)
if show:
    plt.show()
if save:
    plt.savefig("../outputs/"+name + "_%d_hsv_colored_gmm.png"%components)
plt.cla()
print("Done")

import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import colorsys

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch

# Reading Args
def read_args(args):
    name = "peppersc"
    clusters = 2

    if len(args)==2:
        if(args[1].isdigit()):
            clusters = int(args[1])
        else:
            name = args[1]
    elif len(args)==3:
        if(args[1].isdigit() and not args[2].isdigit()):
            clusters = int(args[1])
            name = args[2]
        elif(args[2].isdigit() and not args[1].isdigit()):
            clusters = int(args[2])
            name = args[1]
        else:
            print("Error: Incorrect arguments")
            sys.exit(-1)
    elif len(args)>3:
        print("Error: Too many arguments")
        sys.exit(-1)

    print(("Using %d clusters"%clusters)+" for image "+name)
    return clusters, name

def get_label(algorithm, df, clusters):
    if algorithm=="kmeans":
        model = KMeans(n_clusters = clusters).fit(df)
        label = model.labels_
        centers = model.cluster_centers_
    elif algorithm=="gmm":
        model = GaussianMixture(n_components = clusters).fit(df)
        label = model.predict(df)
        centers = model.means_
    elif algorithm=="birch":
        model = Birch(n_clusters=clusters)
        label = model.predict(df)
        centers = [[0,0] for i in range(clusters)]
        n_label = [0 for i in range(clusters)]
        for it,row in df.iterrows():
            lab = label[it]
            n_label[lab]+=1
            centers[lab] = np.add(centers[lab],row)
        for c, n in zip(centers, n_label):
            c = [i/n for i in c]
    return centers,label

def get_rgb(algorithm, df, clusters, max_x, max_y, show, save):
    # RGB color scheme
    print("\nClustering RGB")
    centers, df['label'] = get_label(algorithm, clusters df[['r','g','b']])
    print("Done")

    print("\nPostprocessing RGB pixels")
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
        plt.savefig("../outputs/"+name + "_%d_rgb_plot_kmeans.png"%clusters)
    plt.cla()
    plt.clf()

    plt.imshow(img_colored)
    if show:
        plt.show()
    if save:
        plt.savefig("../outputs/"+name + "_%d_rgb_colored_kmeans.png"%clusters)
    plt.cla()
    plt.clf()
    print("Done")

def get_hsv(algorithm, df, clusters, max_x, max_y, show, save):
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

    print("\nClustering HS(V)")
    centers, df_hs['label'] = get_label(algorithm, df_hs[['hsy','hsy']])
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
        plt.savefig("../outputs/"+name + "_%d_hsv_plot_kmeans.png"%clusters)
    plt.cla()

    plt.imshow(img_colored)
    if show:
        plt.show()
    if save:
        plt.savefig("../outputs/"+name + "_%d_hsv_colored_kmeans.png"%clusters)
    plt.cla()
    print("Done")

def main(argv):
    clusters, name = read_args(argv)
    show=False
    save=True
    algorithms=["kmeans","gmm","birch"] # Optional: hier_agg

    # Load data

    df = pd.read_csv("../data/"+name+".csv")

    max_x=df["x"].max()
    max_y=df["y"].max()


if __name__ == "__main__":
   main(sys.argv)

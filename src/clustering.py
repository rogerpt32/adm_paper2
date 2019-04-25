import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import colorsys

from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['font.sans-serif'] = ['Console Modern']
rcParams['savefig.format'] = ['pdf']
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# from sklearn.cluster import AgglomerativeClustering
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
    label = []
    centers = []
    if algorithm=="kmeans":
        model = KMeans(n_clusters = clusters).fit(df)
        label = model.labels_
        centers = model.cluster_centers_
    elif algorithm=="gmm":
        model = GaussianMixture(n_components = clusters).fit(df)
        label = model.predict(df)
        centers = model.means_
    elif algorithm=="birch":
        model = Birch(n_clusters=clusters, threshold=0.05).fit(df)
        label = model.labels_
        centers = [[0 for j in range(len(df.columns))] for i in range(clusters)]
        n_label = [0 for i in range(clusters)]
        for it,row in df.iterrows():
            lab = label[it]
            n_label[lab]+=1
            colors=[row[col] for col in df.columns]
            centers[lab] = np.add(centers[lab],colors)
        for it, n in enumerate(n_label):
            centers[it] = [i/n for i in centers[it]]
    return centers,label

def get_rgb(name, algorithm, df, clusters, max_x, max_y, show, save):
    # RGB color scheme
    print("Clustering RGB... ",end='',flush=True)
    centers, df['label'] = get_label(algorithm, df[['r','g','b']], clusters)
    print("Done")

    print("Postprocessing RGB pixels...",end='',flush=True)
    img_colored = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
    colors=[]
    for _,row in df.iterrows():
        x,y,lab = int(row['x']),int(row['y']),int(row['label'])
        img_colored[x][y] = [int(round(c)) for c in centers[lab]]
        colors.append([c/255.0 for c in centers[lab]])
    print("Done")

    print("Generating RGB Plots and images...",end='',flush=True)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(df['r'],df['g'],df['b'],c=colors)
    if show:
        plt.show()
    if save:
        plt.savefig("../outputs/"+name + ("_%d_rgb_plot_"%clusters)+algorithm+".pdf",bbox_inches='tight')
    plt.cla()
    plt.clf()

    plt.imshow(img_colored)
    if show:
        plt.show()
    if save:
        # plt.savefig("../outputs/"+name + ("_%d_rgb_colored_"%clusters)+algorithm+".png")
        plt.imsave("../outputs/"+name + ("_%d_rgb_colored_"%clusters)+algorithm+".png",img_colored)
    plt.cla()
    plt.clf()
    print("Done")

def get_hsv(name, algorithm, df, clusters, max_x, max_y, show, save):
    # HSV color scheme
    # Mapping the colors
    print("Mapping HS(V) colors in 2D space...",end='',flush=True)
    data_hs = []
    for _,row in df.iterrows():
        x,y,h,s = int(row['x']),int(row['y']),row['h'],row['s']
        hsx = s * np.cos(h*(np.pi*2))
        hsy = s * np.sin(h*(np.pi*2))
        data_hs.append([x,y,hsx,hsy])
    df_hs = pd.DataFrame(data=data_hs, columns=["x","y","hsx","hsy"])
    print("Done")

    print("Clustering HS(V)...",end='',flush=True)
    centers, df_hs['label'] = get_label(algorithm, df_hs[['hsx','hsy']], clusters)
    print("Done")

    print("Postprocessing HS(V) pixels...",end='',flush=True)
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
    for _,row in df_hs.iterrows():
        x,y,lab = int(row['x']),int(row['y']),int(row['label'])
        img_colored[x][y] = [int(round(c)) for c in centers_color[lab]]
        colors_hs.append([c/255.0 for c in centers_color[lab]])
    print("Done")

    print("Generating HS(V) Plots and images...",end='',flush=True)
    plt.cla()
    plt.scatter(df_hs['hsx'],df_hs['hsy'],c=colors_hs)
    if show:
        plt.show()
    if save:
        plt.savefig("../outputs/"+name + ("_%d_hsv_plot_"%clusters)+algorithm+".pdf",bbox_inches='tight')
    plt.cla()

    plt.imshow(img_colored)
    if show:
        plt.show()
    if save:
        # plt.savefig("../outputs/"+name + ("_%d_hsv_colored_"%clusters)+algorithm+".png")
        plt.imsave("../outputs/"+name + ("_%d_hsv_colored_"%clusters)+algorithm+".png",img_colored)
    plt.cla()
    print("Done")

def main(argv):
    print("#############################################################")
    clusters, name = read_args(argv)
    show=False
    save=True
    algorithms=["kmeans","gmm","birch"] # Optional: hier_agg

    # Load data

    df = pd.read_csv("../data/"+name+".csv")

    max_x=df["x"].max()
    max_y=df["y"].max()

    for algor in algorithms:
        print("=============================================================")
        print("Algoritm: "+algor)
        print("-------------------------------------------------------------")
        print("RGB:\n")
        get_rgb(name,algor,df,clusters,max_x,max_y,show,save)
        print("-------------------------------------------------------------")
        print("HSV:\n")
        get_hsv(name,algor,df,clusters,max_x,max_y,show,save)
    print("=============================================================")
    print("#############################################################")


if __name__ == "__main__":
   main(sys.argv)

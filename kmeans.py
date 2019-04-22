from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters = 3).fit(df[["r","g","b"]])

print(kmeans.cluster_centers_)
centers = kmeans.cluster_centers_

img_colored = [[0 for i in range(len(img[0]))] for j in range(len(img))]
for x in range(len(img)):
    for y in range(len(img[0])):
        [center] = kmeans.predict([img[x,y]])
        img_colored[x][y] = [int(round(c)) for c in centers[center]]

# print(img_colored)
plt.imshow(img_colored)
plt.show()

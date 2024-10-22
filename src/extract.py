# from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pandas as pd
import numpy as np
import colorsys
import sys

filename = "baboon.png"

if len(sys.argv)==2:
    filename = sys.argv[1]
elif len(sys.argv)>2:
    print("Error: Too many arguments")
    sys.exit(-1)

filename = filename.split("/")[-1] #deletes path if needed

name = filename.split(".")[0]

img = mpimg.imread("../imgs/"+filename)

print("Extracting features of "+ filename)
data = []
for x in range(len(img)):
    for y in range(len(img[0])):
        [r,g,b] = img[x,y]
        if not type(r) is np.uint8:
            r=int(round(r*255))
        if not type(g) is np.uint8:
            g=int(round(g*255))
        if not type(b) is np.uint8:
            b=int(round(b*255))
        (h,s,v) = colorsys.rgb_to_hsv(r,g,b)
        data.append([x,y,r,g,b,h,s,v])

df = pd.DataFrame(data=data, columns=["x","y","r","g","b","h","s","v"])
df.to_csv("../data/"+name+".csv")
print("Features stored at \""+"../data/"+name+".csv"+"\"")

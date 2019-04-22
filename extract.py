# from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pandas as pd
import numpy as np
import colorsys

filename = "peppersc.jpg"
name = filename.split(".")[0]

img = mpimg.imread("imgs/"+filename)
# plt.imshow(img)
# plt.show()
data = []
for x in range(len(img)):
    for y in range(len(img[0])):
        [r,g,b] = img[x,y]
        (h,s,v) = colorsys.rgb_to_hsv(r,g,b)
        data.append([x,y,r,g,b,h,s,v])
df = pd.DataFrame(data=data, columns=["x","y","r","g","b","h","s","v"])
df.to_csv(name+".csv")

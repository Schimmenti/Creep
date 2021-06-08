import numpy as np
import pandas as pd
import argparse

def center(z):
    mask1 = (z==-1).sum(axis=0)
    mask2 = (z==-1).sum(axis=1)
    max1 = mask1.max()
    max2 = mask2.max()
    boundary1 = np.argwhere(mask1 == max1).flatten()
    boundary2 = np.argwhere(mask2==max2).flatten()
    lb1 = np.argmax(np.diff(boundary1))
    lb2 = np.argmax(np.diff(boundary2))
    minb_1, maxb_1 = boundary1[lb1],boundary1[lb1+1] #lateral figure
    minb_2, maxb_2 = boundary2[lb2],boundary2[lb2+1]
    xc = (minb_1+maxb_1)/2
    yc = (minb_2+maxb_2)/2
    radius_x = (maxb_1-minb_1)/2
    radius_y = (maxb_2-minb_2)/2
    return xc,yc,radius_x,radius_y


parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--discretization', type=int, default=700)
args = parser.parse_args()
filename = args.input
out_filename = args.output
n_angles = args.discretization

data = pd.read_csv(filename, header=None).values
xc,yc,radius_x,radius_y = center(data)


import numpy as np
import pandas as pd
import argparse
import pickle
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
parser.add_argument('--initial_time',type=int, default=0)
args = parser.parse_args()
filename = args.input
out_filename = args.output
n_bins = args.discretization+1
t0 = args.initial_time

data = pd.read_csv(filename, header=None).values
xc,yc,radius_x,radius_y = center(data)

#polar coordinates
nrow=data.shape[0]
ncol=data.shape[1]
xcoord = np.arange(0,ncol).reshape(1,-1)
ycoord = np.arange(0,nrow).reshape(-1,1)
th=np.arctan2(ycoord-yc,xcoord-xc)
rr=np.sqrt(np.power((ycoord-yc),2) + np.power((xcoord-xc),2))
#pandas dataframe with triples (th,R,t)
table = pd.DataFrame(np.concatenate([th.reshape(-1,1),rr.reshape(-1,1),data.reshape(-1,1)],axis=1),columns=['th','R','t'])
table = table[table['t'] != -1]
table.sort_values('t', inplace=True)
table['t'] = table['t'].values.astype('int')
table.reset_index(inplace=True)
table.drop('index',axis=1, inplace=True)
max_t = int(table['t'].values.max())

#discretize angles: th -> thi which is an index between 0 (-\pi) and discretization (\pi)
bins = np.linspace(-np.pi, np.pi, n_bins)
digits = np.digitize(table['th'], bins)-1
table['thi'] = digits


interface = {}  #np.zeros((max_t-t0+1,args.discretization))
for t in range(t0, max_t+1):
    if(t % 2000 == 0):
        print("Status: %2.1f%%" % (100*(t-t0)/(max_t+1-t0)))
    z = table[table['t'] <= t]
    rs = z.groupby('thi')['R'].max()
    interface[t] = np.zeros(args.discretization)
    interface[t][rs.index] = rs.values
with open(filename, 'wb') as f:
    pickle.dump(interface,f)
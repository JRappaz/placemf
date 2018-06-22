import sys
import scipy.io
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def flat(x,y):
  return (y+1)*3+(x+1)

with open('data/.csv') as f:
  head = f.readlines()

nb_users = 1166924

adjacency = {}

grid = np.zeros([1001,1001])

f = open('tiles_adjacency.csv','w')

for i,h in enumerate(head):
  la = h.strip().split(',')
  ts    = int(la[0])
  user  = int(la[1])
  x     = int(la[2])
  y     = int(la[3])
  color = int(la[4])

  vec = [0]*14
  vec[0] = ts
  vec[1] = user
  vec[2] = x
  vec[3] = y
  vec[4] = color

  for m in range(-1,2):
    if x+m<0 or x+m>1000:
      continue
    for n in range(-1,2):
      if y+n<0 or y+n>1000:
        continue
      if grid[int(x+m),int(y+n)]==user:
        continue

      if grid[x+m,y+n]!=0:
        vec[5+flat(m,n)] = int(grid[int(x+m),int(y+n)])

  grid[int(x),int(y)] = user

  if i%10000==0:
    print(i)

  f.write( str(vec[0]) )
  for el in vec[1:]:
    f.write( ',' + str(el) )
  f.write( '\n' )

f.close()

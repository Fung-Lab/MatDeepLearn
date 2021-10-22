import ase
import os
from ase.io import read
import numpy as np
import csv

mofs = ase.io.read('opt-geometries.xyz',index=':')
refcodes = np.genfromtxt('opt-refcodes.csv',delimiter=',',dtype=str)
properties = np.genfromtxt('opt-bandgaps.csv',delimiter=',',dtype=str)
print(len(mofs), len(refcodes), properties.shape)

if not os.path.exists('MOF_data'):
  os.mkdir('MOF_data')
count=0
targets=[]
for i in range(0, len(refcodes)):
  ase.io.write(os.path.join('MOF_data',str(refcodes[i])+'.json'), mofs[i])
  targets.append([str(refcodes[i]), properties[i+1,1]])
  count=count+1
with open(os.path.join('MOF_data',"targets.csv"), 'w', newline='') as f:
  wr = csv.writer(f)
  wr.writerows(targets)        
print(count)
  



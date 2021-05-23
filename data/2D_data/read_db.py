import numpy as np
import os
import csv
import ase.db
import ase.io

if not os.path.exists('2D_data'):
  os.mkdir('2D_data')

# Connect to database
db = ase.db.connect('c2db.db')

rows = db.select(selection='workfunction')
count=0
out=[]
for row in rows:
  atoms=row.toatoms()
  out.append(row.workfunction)
  ase.io.write('2D_data/'+str(count)+'.json', atoms)
  count=count+1
print(count)

with open('2D_data/targets.csv', 'w') as f:
  for i in range(0, count):
    f.write(str(i)+','+str(out[i]) + '\n')
from pymatgen import MPRester
from pymatgen.io.cif import CifWriter
import csv
import sys
import os

print('start')

###need 2 files: csv containing materials project IDs and atom_init.json (generic)

###get API key from materials project login dashboard online
API_KEY='aB1VM67J38kTL0Zy'
mpr = MPRester(API_KEY)

###open file containing IDs
f=open('mp-ids-46744.csv')
csvfile = csv.reader(f, delimiter=',')
materials_ids = list(csvfile)
f.close()

print(len(materials_ids))

if not os.path.exists('bulk_data'):
  os.mkdir('bulk_data')

retries=5
count=0
write_prop=open('bulk_data/targets.csv', 'w')
out=[]
for material_id in materials_ids:

  #print(material_id[0])
  try_count=0
  success=0
  while try_count<retries and success==0:
    try:
      out_temp=mpr.query(criteria={"task_id": material_id[0]}, properties=['structure', 'formation_energy_per_atom'], max_tries_per_chunk=10, chunk_size=0)
    except Exception as e:
      print(e)
    try_count=try_count+1

  if len(out_temp) != 0 :
    out.append(out_temp[0])
    write_prop.write(str(count)+','+str(out[count]['formation_energy_per_atom']) + '\n')
    write_cif=CifWriter(out_temp[0]['structure'])
    write_cif.write_file('bulk_data/'+str(count)+'.json')
    
    count=count+1
    if count % 50 ==0:
      print(count)
      
write_prop.close()

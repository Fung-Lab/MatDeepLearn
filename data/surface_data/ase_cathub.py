import requests
import pprint
import sys
import string
import json
import io
import os
import copy
import numpy as np
import ase.io
import ase.calculators.singlepoint
import pickle

GRAPHQL = 'http://api.catalysis-hub.org/graphql'

def fetch(query):
    return requests.get(
        GRAPHQL, {'query': query}
    ).json()['data']
    
def reactions_from_dataset(pub_id, page_size=10):
    reactions = []
    has_next_page = True
    start_cursor = ''
    page = 0
    while has_next_page:
        data = fetch("""{{
      reactions(pubId: "{pub_id}", first: {page_size}, after: "{start_cursor}") {{
        totalCount
        pageInfo {{
          hasNextPage
          hasPreviousPage
          startCursor
          endCursor 
        }}  
        edges {{
          node {{
            Equation
            reactants
            products
            sites
            reactionEnergy
            reactionSystems {{
              name
              systems {{
                energy
                InputFile(format: "json")
              }}
            }}  
          }}  
        }}  
      }}    
    }}""".format(start_cursor=start_cursor,
                 page_size=page_size,
                 pub_id=pub_id,
                ))
        has_next_page = data['reactions']['pageInfo']['hasNextPage']
        start_cursor = data['reactions']['pageInfo']['endCursor']
        page += 1
        print(has_next_page, start_cursor, page_size * page, data['reactions']['totalCount'])
        reactions.extend(map(lambda x: x['node'], data['reactions']['edges']))

    return reactions

def aseify_reactions(reactions):
    for i, reaction in enumerate(reactions):
        for j, _ in enumerate(reactions[i]['reactionSystems']): 
            with io.StringIO() as tmp_file:
                system = reactions[i]['reactionSystems'][j].pop('systems')
                tmp_file.write(system.pop('InputFile'))
                tmp_file.seek(0)
                atoms = ase.io.read(tmp_file, format='json')
            calculator = ase.calculators.singlepoint.SinglePointCalculator(
                atoms,
                energy=system.pop('energy')
            )
            atoms.set_calculator(calculator)
            reactions[i]['reactionSystems'][j]['atoms'] = atoms
        # flatten list further into {name: atoms, ...} dictionary
        reactions[i]['reactionSystems'] = {x['name']: x['atoms']
                                          for x in reactions[i]['reactionSystems']}


reactions = reactions_from_dataset("MamunHighT2019")
aseify_reactions(reactions)

with open('cathub_dump.pkl', 'wb') as f:
    pickle.dump(reactions, f)
#with open('cathub_dump.pkl', 'rb') as f:
#    reactions = pickle.load(f)    

print(len(reactions))
equations=['0.5H2(g) + * -> H*','H2O(g) - H2(g) + * -> O*','H2O(g) - 0.5H2(g) + * -> OH*','H2O(g) + * -> H2O*','CH4(g) - 2.0H2(g) + * -> C*','CH4(g) - 1.5H2(g) + * -> CH*','CH4(g) - H2(g) + * -> CH2*','CH4(g) - 0.5H2(g) + * -> CH3*','0.5N2(g) + * -> N*', 
'0.5H2(g) + 0.5N2(g) + * -> NH*','H2S(g) - H2(g) + * -> S*','H2S(g) - 0.5H2(g) + * -> SH*']
key_name=['Hstar','Ostar', 'OHstar', 'H2Ostar', 'Cstar', 'CHstar', 'CH2star', 'CH3star', 'Nstar', 'NHstar', 'Sstar', 'SHstar']

os.system('mkdir ads_data')

count=0
energies=[]
for i in range(0, len(equations)):
  for j in range(0, len(reactions)):
    try:
      if reactions[j]['Equation'].find(equations[i]) != -1:  
        if reactions[j]['sites'].find('top|') != -1:
          ase.io.write('ads_data/'+str(count)+'.json', reactions[j]['reactionSystems'][key_name[i]])
          count=count+1      
          energies.append(reactions[j]['reactionEnergy'])  
        elif reactions[j]['sites'].find('bridge|') != -1:
          ase.io.write('ads_data/'+str(count)+'.json', reactions[j]['reactionSystems'][key_name[i]])
          count=count+1      
          energies.append(reactions[j]['reactionEnergy'])
        elif reactions[j]['sites'].find('hollow|') != -1:
          ase.io.write('ads_data/'+str(count)+'.json', reactions[j]['reactionSystems'][key_name[i]])
          count=count+1      
          energies.append(reactions[j]['reactionEnergy'])
    except Exception as e:
      print(e)

print(count)

with open('ads_data/targets.csv', 'w') as f:
  for i in range(0,len(energies)):
    f.write(str(i)+','+str(energies[i]) + '\n')


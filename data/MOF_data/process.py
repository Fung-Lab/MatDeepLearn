import os
import csv
import json
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

# Read in QMOF data
with open("qmof.json") as f:
    qmof = json.load(f)
qmof_df = pd.json_normalize(qmof).set_index("qmof_id")

with open("qmof_structure_data.json") as f:
    qmof_struct_data = json.load(f)

# Make MOF_data folder
if not os.path.exists("MOF_data"):
    os.mkdir("MOF_data")

# Write out data
targets = []
for entry in qmof_struct_data:
    qmof_id = entry["qmof_id"]
    print(f"Writing {qmof_id}")
    mof = AseAtomsAdaptor().get_atoms(Structure.from_dict(entry["structure"]))
    write(os.path.join("MOF_data", f"{qmof_id}.json"), mof)
    targets.append([qmof_id, qmof_df.loc[qmof_id]["outputs.pbe.bandgap"]])

with open(os.path.join("MOF_data", "targets.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerows(targets)

print(len(targets))

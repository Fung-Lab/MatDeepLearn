
Five datasets were used in benchmarking, and were chosen to reflect a variety of different materials and properties found in computational studies. They are:

1. A 3D bulk crystal structure dataset obtained from [Materials Project](https://www.materialsproject.org), for prediction of formation energies. 

2. A 3D porous material dataset of MOFs obtained from Rosen et al. at [QMOF](https://github.com/arosen93/QMOF), for prediction of band gaps.

3. A metal alloy surface dataset obtained from Mamun et al. at [CatHub](https://www.catalysis-hub.org/), for prediction of adsorption energies.

4. A 2D material dataset obtained from Haastrup at al. at [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html), for prediction of workfunctions.

5. A 0D sub-nanometer Pt cluster dataset from Fung et al., for prediction of total energies. Provided here as pt_data.tar.gz as well as test_data.tar.gz as a subset.

Datasets include the individual structure files, encoded as an ASE .json file, a dictionary of elemental properties in atom_dict.json, and the prediction targets in targets.csv. Unzip the datasets via the command "tar -xvf XXX.tar.gz" to use. We cannot redistribute all of the referenced datasets here, please follow the links for instructions to download. We will include additional datasets of our own in the near future.

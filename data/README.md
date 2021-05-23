
Five datasets were used in benchmarking in https://chemrxiv.org/articles/preprint/Benchmarking_Graph_Neural_Networks_for_Materials_Chemistry/13615421, and were chosen to reflect a variety of different materials and properties found in computational studies. They are:

1. A 3D bulk crystal structure dataset obtained from [Materials Project](https://materialsproject.org/open), for prediction of formation energies (bulk_dataset).

2. A 3D porous material dataset of MOFs obtained from Rosen et al. from the [QMOF database](https://github.com/arosen93/QMOF), for prediction of band gaps (MOF_dataset).

3. A metal alloy surface dataset obtained from Mamun et al., for prediction of adsorption energies from [CatHub](https://www.catalysis-hub.org/) (surface_dataset).

4. A 2D material dataset obtained from Haastrup at al., for prediction of workfunctions from [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html) (2D_dataset).

5. A 0D sub-nanometer Pt cluster dataset from [Fung et al.](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b11968), for prediction of total energies, provided here in full (pt_dataset).

Datasets include the individual structure files, encoded as an ASE .json file, a dictionary of elemental properties in atom_dict.json, and the prediction targets in targets.csv. Unzip the datasets via the command "tar -xvf XXX.tar.gz" to use. We cannot redistribute all of the referenced datasets here, please follow the links for instructions to download. We will include additional datasets of our own in the near future.

import os
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata

import torch
# from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops

################################################################################
# Data processing
################################################################################

# basic train, val, test split
def split_data(
    dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=np.random.randint(1, 1e6),
    save=False,
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")


# basic CV split
def split_data_CV(dataset, num_folds=5, seed=np.random.randint(1, 1e6), save=False):
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for i in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    print("fold length :", fold_length, "unused length:", unused_length, "seed", seed)
    return cv_dataset[0:num_folds]


# process data, uses the dataset class from pytorch/pytorch geometric
class StructureDataset(InMemoryDataset):
    def __init__(self, data_path, save_dir, params, transform=None, pre_transform=None):
        self.params = params
        self.data_path = data_path
        self.save_dir = save_dir
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.save_dir, "processed")

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        # for file in glob.glob(self.processed_dir+"/data*.pt"):
        #  file_names.append(file)
        return file_names

    def process(self):

        # begin processing data
        print("processing data to: " + self.processed_dir)
        assert os.path.exists(self.data_path), (
            "data_path not found in " + self.data_path
        )
        target_property_file = os.path.join(self.data_path, self.params.target_path)
        assert os.path.exists(target_property_file), (
            "targets not found in " + target_property_file
        )
        with open(target_property_file) as f:
            reader = csv.reader(f)
            target_data = [row for row in reader]

        dictionary_file = os.path.join(self.data_path, self.params.dictionary_path)
        assert os.path.exists(dictionary_file), "atom dictionary not found"
        atom_dictionary = get_dictionary(dictionary_file)
        distance_gaussian = GaussianSmearing(
            self.params.min_radius, self.params.max_radius, self.params.gaussians
        )

        data_list = []
        
        length = []
        elements = []
        ase_crystal= []
        for index in range(0, len(target_data)):
            structure_id = target_data[index][0]
            # read in structure file using ase
            ase_temp = ase.io.read(
                os.path.join(self.data_path, structure_id + "." + self.params.format))
            ase_crystal.append(ase_temp)
            # compile structure sizes (# of atoms) and elemental compositions
            length.append(len(ase_temp))
            elements.append(list(set(ase_temp.get_chemical_symbols())))
            
        n_atoms_max = max(length)
        species = list(set(sum(elements, [])))
        species.sort()
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            len(species),
        )
        
        if self.params.voronoi == "True":
            from pymatgen.core.structure import Structure
            from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
            from pymatgen.io.ase import AseAtomsAdaptor        
            Converter = AseAtomsAdaptor()
                        
        if self.params.extra_features == "True":
            from dscribe.descriptors import CoulombMatrix, SOAP, MBTR, EwaldSumMatrix, SineMatrix        
            make_feature_SOAP = SOAP(
                species=species,
                rcut=self.params.SOAP_rcut,
                nmax=self.params.SOAP_nmax,
                lmax=self.params.SOAP_lmax,
                sigma=self.params.SOAP_sigma,
                periodic=True,
                sparse=False,
                average="inner",
                rbf="gto",
                crossover=False,
            )
            make_feature_SM = SineMatrix(
                n_atoms_max=n_atoms_max,
                permutation="eigenspectrum",
                sparse=False,
                flatten=True,
            )         
        
        ###loop over all structures
        for index in range(0, len(ase_crystal)):
                
            structure_id = target_data[index][0]
            data = Data(
                structure_id=structure_id,
            )
            # set targets
            if len(target_data[index]) > 1:
                target = target_data[index][1:]
                y = torch.Tensor(np.array(target, dtype=np.float32))
                data.y = y
            else:
                target = np.empty((max(map(len, target_data))-1,))
                target[:] = np.nan
                y = torch.Tensor(target)
                data.y = y

            # obtain distance matrix with ase
            distance_matrix = ase_crystal[index].get_all_distances(mic=True)
            
            # create graph from distance matrix
            distance_matrix_trimmed = threshold_sort(
                distance_matrix,
                self.params.max_radius,
                self.params.max_neighbors,
                adj=False,
            )
            distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
            out = dense_to_sparse(distance_matrix_trimmed)
            edge_index = out[0]
            edge_weight = out[1]
            
            # edge weights defined here as inverse of distance
            #edge_weight = 1 - (edge_weight / self.params.max_radius)
            #edge_index, edge_weight = add_self_loops(
            #    edge_index, edge_weight, num_nodes=len(ase_crystal[index]), fill_value=1
            #)
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal[index]), fill_value=0
            )
                        
            edge_attr = distance_gaussian(edge_weight)
            edge_attr = edge_attr.float()
            # print(edge_index.shape, edge_weight.shape, edge_attr.shape)

            # atom features current use elemental properties from atom dictionary file
            atom_fea = np.vstack(
                [
                    atom_dictionary[str(ase_crystal[index].get_atomic_numbers()[i])]
                    for i in range(len(ase_crystal[index]))
                ]
            ).astype(float)

            # x: atom(node) features, y:target, pos:cartesian positions, z:atomic numbers, u:state/global feature, cell:lattice parameters
            x = torch.Tensor(atom_fea)
            pos = torch.Tensor(ase_crystal[index].get_positions())
            z = torch.LongTensor(ase_crystal[index].get_atomic_numbers())
            cell = ase_crystal[index].get_cell_lengths_and_angles()
            cell = torch.Tensor(cell[np.newaxis, ...])

            data.x = x
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.edge_weight = edge_weight
            data.pos = pos
            data.z = z
            data.cell = cell
            data.atoms = ase_crystal[index]

            ###placeholder for state feature
            u = np.zeros((3))
            u = torch.Tensor(u[np.newaxis, ...])
            data.u = u

            # get graphs vased on voronoi connectivity; todo: also get voronoi features
            # avoid use for the time being until a good approach is found
            if self.params.voronoi == "True":
                
                pymatgen_crystal = Converter.get_structure(ase_crystal[index])
                #double check if cutoff distance does anything
                Voronoi = VoronoiConnectivity(pymatgen_crystal, cutoff=self.params.max_radius)
                connections = Voronoi.max_connectivity
                
                distance_matrix_voronoi = threshold_sort(
                    connections,
                    9999,
                    self.params.max_neighbors,
                    reverse=True,
                    adj=False,
                )
                distance_matrix_voronoi = torch.Tensor(distance_matrix_voronoi)

                out = dense_to_sparse(distance_matrix_voronoi)
                edge_index_voronoi = out[0]
                edge_weight_voronoi = out[1]

                edge_attr_voronoi = distance_gaussian(edge_weight_voronoi)
                edge_attr_voronoi = edge_attr_voronoi.float()

                data.edge_index_voronoi = edge_index_voronoi
                data.edge_weight_voronoi = edge_weight_voronoi
                data.edge_attr_voronoi = edge_attr_voronoi

            # get fully connected graph
            if self.params.fullconn == "True":
                distance_matrix_full = threshold_sort(
                    distance_matrix, 9999, 9999, adj=False
                )

                distance_matrix_full = torch.Tensor(distance_matrix_full)
                out = dense_to_sparse(distance_matrix_full)
                edge_index_full = out[0]
                edge_weight_full = out[1]
                edge_index_full, edge_weight_full = add_self_loops(
                    edge_index_full,
                    edge_weight_full,
                    num_nodes=len(ase_crystal[index]),
                    fill_value=0,
                )
                edge_attr_full = distance_gaussian(edge_weight_full)
                edge_attr_full = edge_attr_full.float()

                data.edge_index_full = edge_index_full
                data.edge_weight_full = edge_weight_full
                data.edge_attr_full = edge_attr_full

            # makes SOAP and SM features from dscribe
            if self.params.extra_features == "True":                            
            
                features_SM = make_feature_SM.create(ase_crystal[index])
                data.extra_features_SM = torch.Tensor(features_SM) 

                features_SOAP = make_feature_SOAP.create(ase_crystal[index])
                data.extra_features_SOAP = torch.Tensor(features_SOAP)

                if index == 0:      
                    print(
                        "SM length: ",
                        features_SM.shape,
                        "SOAP length: ",
                        features_SOAP.shape,
                    )

            if index % 500 == 0:
                print("data processed: ", index)
                #if index == 0:
                #    print(data)
  
            data_list.append(data)

            # torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(index)))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    """
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    """


# selects edges with distance threshold and limited number of neighbors
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr


# slightly edited version from pytorch geometric
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


# obtain dictionary file for elemental features
def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


################################################################################
#  Transforms
################################################################################

class Get_Y(object):
    def __init__(self, index=0):
        self.index = index
        
    def __call__(self, data):
        # Specify target.
        data.y = data.y[self.index]
        return data

class Get_Voronoi(object):
    def __call__(self, data):
        data.edge_index = data.edge_index_voronoi
        data.edge_weight = data.edge_weight_voronoi
        data.edge_attr = data.edge_attr_voronoi
        return data


class Get_Full(object):
    def __call__(self, data):
        data.edge_index = data.edge_index_full
        data.edge_weight = data.edge_weight_full
        data.edge_attr = data.edge_attr_full
        return data

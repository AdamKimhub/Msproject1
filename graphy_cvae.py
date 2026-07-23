import numpy as np
import random

import torch
from torch_geometric.data import Data

from pymatgen.core import Structure, PeriodicSite, DummySpecie, Element
from pymatgen.core.periodic_table import Element as PMGElement
from pymatgen.analysis.local_env import MinimumDistanceNN
# from mp_api.client import MPRester

masking_dict = {
    "GaSe": {"to_sub":["In", "S"],  "possible_defects": [3, 7, 10, 14, 18]},
    "InSe": {"to_sub":["Ga", "S"],  "possible_defects": [3, 7, 10, 14, 18]},
    "BN"  : {"to_sub":["C"],        "possible_defects": [3, 6,  9, 12, 16]},
    "P"   : {"to_sub":["N"],        "possible_defects": [3, 7, 10, 14, 18]},
    "WSe2": {"to_sub":["Mo", "S"],  "possible_defects": [1,2,3,4, 9, 14, 19, 24]},
    "MoS2": {"to_sub":["W", "Se"],  "possible_defects": [1,2,3,4, 9, 14, 19, 24]}
}

the_materials_list = list(masking_dict.keys())

# ==============================
# FULL DEFECTIVE STRUCTURE
# ==============================

def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}


def align_to_reference_lattice(defective_struct, reference_struct):
    if not np.allclose(defective_struct.lattice.matrix,
                       reference_struct.lattice.matrix,
                       atol=1e-6):
        frac_coords = reference_struct.lattice.get_fractional_coords(defective_struct.cart_coords)
        frac_coords = np.mod(frac_coords, 1.0)
        return Structure(reference_struct.lattice,
                         defective_struct.species,
                         frac_coords,
                         coords_are_cartesian=False)
    return defective_struct


def get_full_defective(reference_struct, defective_struct):
    # mindnn = MinimumDistanceNN()
    # Align the lattice
    if reference_struct.lattice != defective_struct.lattice:
        defective_struct = align_to_reference_lattice(defective_struct, reference_struct)
    else:
        pass

    # struct to dict
    defective_dict = struct_to_dict(defective_struct)
    reference_dict = struct_to_dict(reference_struct)

    # Get lattice of defective structure
    structure_lattice = defective_struct.lattice

    # List to add all defect sites
    defective_structure_list = []

    # Dictionary to hold properties of each site
    defects_properties = {}

    ref_index = 0

    for ref_coord, ref_site in reference_dict.items():
        # Use the reference coordinates to get the defective site
        ref_index = ref_index + 1

        def_site = defective_dict.get(ref_coord)

        if def_site:  # The site is found in both the reference structure and the defective structure
            # But are the species the same?
            ref_specie = ref_site.specie
            def_specie = def_site.specie
            if ref_specie != def_specie:  # Substitution
                # Add site to defects list
                defective_structure_list.append(def_site)

                # Get atomic number change and defect type
                add_property = {
                    "original_Z":ref_specie.Z,
                    "new_Z": def_specie.Z,

                    "original_en": ref_specie.X,
                    "new_en": def_specie.X,

                    "original_ar": ref_specie.atomic_radius,
                    "new_ar": def_specie.atomic_radius,

                    "original_row": ref_specie.row,
                    "new_row": def_specie.row,

                    "original_group": ref_specie.group,
                    "new_group": def_specie.group,

                    "original_max_os": max(ref_specie.common_oxidation_states),
                    "new_max_os": max(def_specie.common_oxidation_states) if def_specie.common_oxidation_states else 0,

                    "original_ef": ref_specie.electron_affinity,
                    "new_ef": def_specie.electron_affinity,

                    "vacancy_defect": 0.0,
                    "substitution_defect": 1.0,
                    "normal_site":0.0,
                }

                defects_properties[def_site] = add_property
            else: # Normal site
                defective_structure_list.append(ref_site)

                add_property = {
                    "original_Z":ref_specie.Z,
                    "new_Z": ref_specie.Z,

                    "original_en": ref_specie.X,
                    "new_en": ref_specie.X,

                    "original_ar": ref_specie.atomic_radius,
                    "new_ar": ref_specie.atomic_radius,

                    "original_row": ref_specie.row,
                    "new_row": ref_specie.row,

                    "original_group": ref_specie.group,
                    "new_group": ref_specie.group,

                    "original_max_os": max(ref_specie.common_oxidation_states),
                    "new_max_os": max(ref_specie.common_oxidation_states),

                    "original_ef": ref_specie.electron_affinity,
                    "new_ef": ref_specie.electron_affinity,

                    "vacancy_defect": 0.0,
                    "substitution_defect": 0.0,
                    "normal_site":1.0
                }

                defects_properties[ref_site] = add_property

    
        else: # the site from ref_structure is not found in defective structure
            # This means that the site is a vacancy site
            # Add site to defective structure
            vacant_site = PeriodicSite(
                species= DummySpecie(),
                coords= ref_coord,
                coords_are_cartesian= False,
                lattice= structure_lattice
                )

            # Add site to defects list
            defective_structure_list.append(vacant_site)

            ref_specie = ref_site.specie

            # Get atomic number change and defect type
            add_property={
                "original_Z":ref_specie.Z,
                "new_Z": 0,

                "original_en": ref_specie.X,
                "new_en": 0.0,

                "original_ar": ref_specie.atomic_radius,
                "new_ar": 0.0,

                "original_row": ref_specie.row,
                "new_row": 0,

                "original_group": ref_specie.group,
                "new_group": 0,

                "original_max_os": max(ref_specie.common_oxidation_states),
                "new_max_os": 0,

                "original_ef": ref_specie.electron_affinity,
                "new_ef": 0.0,
                
                "vacancy_defect": 1.0,
                "substitution_defect": 0.0,
                "normal_site":0.0

            }
            defects_properties[vacant_site] = add_property

    # create a defects structure
    defective_struct = Structure.from_sites(defective_structure_list)

    # Add properties to defects structure
    for a_site in defective_struct.sites:
        if a_site in defects_properties.keys():
            a_site.properties.update(defects_properties[a_site])
        else:
            pass

    return defective_struct

# =====================
# CLOUD STRUCTURRE
# =====================

def get_cloud(full_defective_structure, pristine_structure):

    full_defective_copy = full_defective_structure.copy()
    
    cloud_list = []

    for p,d in zip(pristine_structure.sites, full_defective_copy.sites):
        if p.specie != d.specie:
            cloud_list.append(d)

    cloud_structure = Structure.from_sites(cloud_list)
        
    return cloud_structure

# =================
# NODES
# =================

def get_nodes(full_defective_structure):
    all_sites = full_defective_structure.sites

    nodes = []
    for site in all_sites:
        coords = site.frac_coords.tolist()
        site_features = [
            site.properties["new_Z"]/94,
            site.properties["new_ar"],
            site.properties["new_ef"],
            site.properties["new_en"]/4,
            site.properties["new_group"]/18,
            site.properties["new_max_os"],
            site.properties["new_row"]/9,
            
            site.properties["original_Z"]/94,
            site.properties["original_ar"],
            site.properties["original_ef"],
            site.properties["original_en"]/4,
            site.properties["original_group"]/18,
            site.properties["original_max_os"],
            site.properties["original_row"]/9,

            site.properties["vacancy_defect"],
            site.properties["substitution_defect"],
            # site.properties["normal_site"],
        ]
        nodes.append(coords + site_features)

    # len_nodes = len(nodes)

    # if len_nodes < max_points:
        # Pad with zeros
        # padding_nodes = [[0.0]*len(nodes[0])]*(max_points - len_nodes)
        # nodes.extend(padding_nodes)

    return nodes

# ============================
# EDGES
# ============================

def get_edges(full_defective_structure):
    nn = MinimumDistanceNN()

    edges = []
    from_edges = []
    to_edges = []

    # Get all the defective sites and their indices. Runs once through the structure
    defective_sites = []
    defective_sites_indices = []

    for indx, site in enumerate(full_defective_structure.sites):
        if site.properties["normal_site"] != 1:
            defective_sites.append(site) 
            defective_sites_indices.append(indx)
            
        else:
            continue 

    # Get nearest defect site of every site
    for i, site_i in enumerate(full_defective_structure.sites):
        to_i = []

        dist_to_defect = [site_i.distance(def_site) for def_site in defective_sites]

        sorted_distances = sorted(dist_to_defect)
        len_sorted = len(sorted_distances)

        if sorted_distances[0] == 0.0 and len_sorted > 1:
            min_dist = sorted_distances[1]
        elif sorted_distances[0] != 0.0 and len_sorted > 1:
            min_dist = sorted_distances[0]
        else:
            min_dist  = sorted_distances[0]

        # Which defect site?
        min_idx = dist_to_defect.index(min_dist)

        # focus_defect_site = defective_sites[min_idx]
        focus_defect_index = defective_sites_indices[min_idx]
        to_i.append(focus_defect_index)
        

            
        nn_info = nn.get_nn_info(full_defective_structure, i)
        for info in nn_info:
            nbr_site = info["site"]
            j = int(info["site_index"])  # often available
            to_i.append(j)

        # Remove duplicates in to_i
        unique_j = list(set(to_i))
        from_i = [i] * len(unique_j)
        for a,b in zip(from_i, unique_j):
            from_edges.append(a)
            to_edges.append(b)


    # Create a list of tuples
    edge_pairs = list(zip(from_edges, to_edges))

    new_edge_pairs = []
    for pair in edge_pairs:
        if pair[0] != pair[1]:
            one_pair = frozenset(pair)
        else:
            one_pair = pair
        new_edge_pairs.append(one_pair)

    unique_edge_pairs = list(set(new_edge_pairs))

    from_edges, to_edges = map(list, zip(*unique_edge_pairs))


    edges.append(from_edges)
    edges.append(to_edges)

    return edges

# ====================
# EDGE FEATURES
# ====================

def get_features(edges, full_defective_structure):

    full_defective_sites = full_defective_structure.sites
    a_lat = float(full_defective_structure.lattice.a)

    from_edges = edges[0]
    to_edges = edges[1]

    edge_features = []

    for idx_i, idx_j in zip(from_edges, to_edges):
        site_i = full_defective_sites[idx_i]
        site_j = full_defective_sites[idx_j]

        dist = site_i.distance(site_j)

        cart_i = site_i.coords
        cart_j = site_j.coords
        r_vec = cart_j - cart_i
        r_ij = float(np.linalg.norm(r_vec))

        dist_angstrom = r_ij
        dist_norm = dist
        dist_lattice_units = r_ij/a_lat

        q_i = site_i.properties["new_max_os"]
        q_j = site_j.properties["new_max_os"]

        if site_i.properties["vacancy_defect"] == 1:
            q_i = -q_i

        if site_j.properties["vacancy_defect"] == 1:
            q_j = -q_j

        charge_product = q_i * q_j
        screened_coulomb = (q_i * q_j) / (r_ij) if r_ij > 0 else 0.0

        # Angular factor
        if r_ij > 0:
            cos_theta = r_vec[2]/ r_ij
            angular_factor = 1.0-3.0 * cos_theta ** 2
        else:
            angular_factor = 0.0

        the_features = [
            dist, dist_angstrom, dist_norm, dist_lattice_units,
            charge_product, screened_coulomb,angular_factor
        ]
        edge_features.append(the_features)

    return edge_features

# ==========
#  MASK
# ==========

def get_mask(reference_structure):
    ref_material = reference_structure.reduced_formula
    # len_ref_sites = len(reference_structure)

    positive_mask = np.zeros(119, dtype=bool)
    positive_mask[0] = True   # vacancy always valid

    allowed_subs = masking_dict[ref_material]["to_sub"]

    for i in allowed_subs:
        allowed_elem_z = Element(i).Z
        positive_mask[allowed_elem_z] = True

    # return torch.tensor([positive_mask] * len_ref_sites, dtype=torch.bool)
    return positive_mask

# ====================
# GLOBAL ATTRIBUTES
# ====================

def get_globals(the_material, bgv):
    one_hot = [1 if a_material == the_material else 0 for a_material in the_materials_list]
    global_list = one_hot + [bgv]
    return global_list

# ======================
# NUMBER OF DEFECTS
# ======================

def get_num_defects(the_material):
    poss_ds = masking_dict[the_material]["possible_defects"]
    the_ds = random.choice(poss_ds)
    return the_ds

# =========================
# COMBINE TO GET GRAPHS
# =========================

def get_graphs(material_dataset):
    data_list = []
    dataset_materials = list(material_dataset["dataset_material"].unique())

    ref_structures = [Structure.from_file(f"Final_Dataset/ref_cifs/{dm}.cif") for dm in dataset_materials]
    the_masks      = [get_mask(rs) for rs in ref_structures]
    the_materials  = [dm.split("_")[1] for dm in dataset_materials]
    
    
    for focus_index, the_dataset_material in enumerate(dataset_materials):
        # The data, prstine structure, and mask
        focus_data = material_dataset[material_dataset["dataset_material"] == the_dataset_material]

        ref_structure = ref_structures[focus_index]
        the_mask      = the_masks[focus_index]
        the_material  = the_materials[focus_index]

        for index, row in focus_data.iterrows():
            the_id = row["_id"]
            bgv = row["band_gap_value"]
            
            defective_structure      = Structure.from_file(f"original_dataset/{the_dataset_material}/cifs/{the_id}.cif")
            full_defective_structure = get_full_defective(ref_structure, defective_structure)
            # cloud_structure        = get_cloud(full_defective_structure)

            # the_nodes   = get_nodes(cloud_structure, max_points)
            the_nodes     = get_nodes(full_defective_structure)
            the_edges     = get_edges(full_defective_structure)
            the_features  = get_features(the_edges, full_defective_structure)
            the_condition = get_globals(the_material, bgv)
            
            data = Data(
                x          =torch.tensor(the_nodes, dtype=torch.float),
                edge_index =torch.tensor(the_edges, dtype=torch.long),
                edge_attr  =torch.tensor(the_features, dtype=torch.float),
                u          =torch.tensor(the_condition, dtype=torch.float).unsqueeze(0), 
                mask       =torch.tensor(the_mask, dtype=torch.bool)
                )
        
            data_list.append(data)
        
    return data_list
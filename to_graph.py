# Get dependacies
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from pymatgen.core import Structure, PeriodicSite, DummySpecie
from pymatgen.analysis.local_env import MinimumDistanceNN

def struct_to_dict(structure):
    list_of_sites = structure.sites
    list_of_frac_coords = np.round(structure.frac_coords,3)
    structure_dict = {i: j for i, j in zip(list_of_sites, list_of_frac_coords)}
    return structure_dict

def get_index(ref_struct, a_site):
    for index, site in enumerate(ref_struct.sites):
        if np.array_equal(site.coords, a_site.coords):
            return index


def get_defects_structure(defective_struct, reference_struct):
    copy_defective_struct = defective_struct.copy()
    mindnn = MinimumDistanceNN()
    # struct to dict
    defective_dict = struct_to_dict(copy_defective_struct)
    reference_dict = struct_to_dict(reference_struct)

    # Get lattice of defective structure
    structure_lattice = copy_defective_struct.lattice

    # List to add all defect sites
    defects_list = []

    # Dictionary to hold properties of each defect site
    defects_properties = {} 

    for ref_site, ref_coords in reference_dict.items():
        matching = False
        for def_site, def_coords in defective_dict.items():
            if np.array_equal(ref_coords, def_coords):
                matching = True
                if ref_site.specie != def_site.specie: # Substitution case
                    # Add site to defects list
                    defects_list.append(def_site)

                    # Get atomic number change and defect type
                    add_property = {"original_an":ref_site.specie.Z,
                                    "new_an": def_site.specie.Z,
                                    "an_change": def_site.specie.Z - ref_site.specie.Z,
                                    "vacancy_defect": 0.0,
                                    "substitution_defect": 1.0,
                                    "bonds_broken": 0.0}
                    defects_properties[def_site] = add_property

        if not matching: # Vacancy case
            # Add site to defective structure
            vacant_site = PeriodicSite(
                species= DummySpecie(),
                coords= ref_coords,
                coords_are_cartesian= False, 
                lattice= structure_lattice
                )
            
            # Add site to defects list
            defects_list.append(vacant_site)

            # Get atomic number change and defect type
            add_property={"original_an":ref_site.specie.Z,
                          "new_an": 0,
                          "an_change": 0 - ref_site.specie.Z,
                          "vacancy_defect": 1.0,
                          "substitution_defect": 0.0,
                          "bonds_broken": mindnn.get_cn(reference_struct, get_index(reference_struct, ref_site))}
            defects_properties[vacant_site] = add_property

    # create a defects structure
    defects_struct = Structure.from_sites(defects_list)

    # Add properties to defects structure
    for a_site in defects_struct.sites:
        if a_site in defects_properties.keys():
            a_site.properties.update(defects_properties[a_site])
        else:
            pass

    return defects_struct

def get_nodes_edges(structure):
    sites_list = structure.sites

    # The nodes: These are the sites features
    nodes = []
    for i, site in enumerate(sites_list):
        node_features = [i, site.properties["bonds_broken"], site.properties["original_an"], site.properties["new_an"],
                         site.properties["an_change"], site.properties["vacancy_defect"],
                         site.properties["substitution_defect"]]
        # Node features syntax
        '''[index of site, number of nearset neighbors to site, Z_before defect,
        Z_after defect, Z_change, is site vac_site(1  for yes, 0 for no), is site sub_site(1 for yes, 0 for no)]'''
        nodes.append(node_features)
         

    # The edges
    edges = [] # The sites in relation
    edge_features = [] # The distance between each site

    from_l = []
    to_l = []
    for i, site_i in enumerate(sites_list):
        for j, site_j  in enumerate(sites_list):
            if i != j:
                from_l.append(i)
                to_l.append(j)
                dist = site_i.distance(site_j)
                edge_features.append([dist])
    edges.append(from_l)
    edges.append(to_l)
    return nodes, edges, edge_features


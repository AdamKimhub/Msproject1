# Get dependacies
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from pymatgen.core import Structure, PeriodicSite, DummySpecie
from pymatgen.analysis.local_env import MinimumDistanceNN

def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}

def fe_site(original, new):
    # from elements.csv
    element_pot = element_pot = {"Mo":-10.9332, "S":-4.127, "W":-13.0106, "Se":-3.489,
                   "B":-6.704, "N":-8.324, "Ga":-3.03, "In":-2.715,
                   "P":-5.362, "V":-8.992, "O":-4.938, "C":-9.226}
    
    if new == 0: # For vcancy
        fe_defect = element_pot[original]* -1

    else: # For substitution
        fe_defect = (element_pot[original]* -1) + element_pot[new]
        
    return fe_defect

def get_defects_structure(defective_struct, reference_struct):
    mindnn = MinimumDistanceNN()
    # struct to dict
    defective_dict = struct_to_dict(defective_struct)
    reference_dict = struct_to_dict(reference_struct)

    # Get lattice of defective structure
    structure_lattice = defective_struct.lattice

    # List to add all defect sites
    defects_list = []

    # Dictionary to hold properties of each defect site
    defects_properties = {} 

    for ref_coord, ref_site in reference_dict.items():
        # Use the reference coordinates to get the defective site
        def_site = defective_dict.get(ref_coord)

        if def_site:  # The site is found in both the reference structure and the defective structure
            # But are the species the same?
            if ref_site.specie != def_site.specie:  # Substitution
                # Add site to defects list
                defects_list.append(def_site)

                # Get atomic number change and defect type
                add_property = {"original_an":ref_site.specie.Z,
                                "new_an": def_site.specie.Z,
                                "an_change": def_site.specie.Z - ref_site.specie.Z,
                                "vacancy_defect": 0.0,
                                "substitution_defect": 1.0,
                                "bonds_broken": 0.0,
                                "site_fe": fe_site(ref_site.species_string,def_site.species_string)}
                defects_properties[def_site] = add_property

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
            defects_list.append(vacant_site)

            # Get atomic number change and defect type
            add_property={"original_an":ref_site.specie.Z,
                          "new_an": 0,
                          "an_change": 0 - ref_site.specie.Z,
                          "vacancy_defect": 1.0,
                          "substitution_defect": 0.0,
                          "bonds_broken": mindnn.get_cn(reference_struct, reference_struct.sites.index(ref_site)),
                          "site_fe": fe_site(ref_site.species_string,0)}
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
        node_features = [i, site.properties["bonds_broken"], site.properties["original_an"], 
                         site.properties["new_an"], site.properties["an_change"], 
                         site.properties["vacancy_defect"], site.properties["substitution_defect"], 
                         site.properties["site_fe"]]
        # Node features syntax
        nodes.append(node_features)
         

    # The edges
    edges = [] # The sites in relation
    edge_features = [] # The distance between each site

    from_e = []
    to_e = []
    
    for i, site_i in enumerate(sites_list):
        for j, site_j  in enumerate(sites_list):
            from_e.append(i)
            to_e.append(j)
            # Get distance between sites
            dist = site_i.distance(site_j)

            # Are the defects the same or different
            if site_i.properties["an_change"] == site_j.properties["an_change"]:
                same_diff = 1
            else:
                same_diff = 0

            # What is the site_fe difference
            site_fe_diff = np.abs(site_i.properties["site_fe"] - site_j.properties["site_fe"])

            edge_features.append([dist,same_diff,site_fe_diff])
    edges.append(from_e)
    edges.append(to_e)
    return nodes, edges, edge_features
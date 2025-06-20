# Get dependacies
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from pymatgen.core import Structure, PeriodicSite, DummySpecie, Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import MinimumDistanceNN
import json
from mp_api.client import MPRester
import config
API_KEY = config.API_KEY

def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}

def get_formation(element, API_KEY):
    with MPRester(API_KEY) as mpr:
        results = mpr.materials.summary.search(
            elements=[element],
            num_elements=1,
            fields= ["energy_per_atom"]
        )
        forms_list = [result.energy_per_atom for result in results]
        avg_formation_energy = np.mean(forms_list)
        
    return avg_formation_energy
    

def get_from_json(element, API_KEY):
    with open("./test.json", "r") as f:
        try:
            the_dict = json.load(f)
            if element in the_dict:
                to_return = the_dict[element]

            else:
                to_return = get_formation(element, API_KEY)
                the_dict[element] = to_return
                with open("./test.json", "w") as f:
                    json.dump(the_dict, f)

        except:
            the_dict = {}
            with open("./test.json", "a") as f:
                to_return = get_formation(element, API_KEY)
                the_dict[element] = to_return
                json.dump(the_dict, f)

        return to_return

def fe_site(original, new):
    if new == 0: # For vcancy
        fe_defect = get_from_json(original, API_KEY) * -1

    else: # For substitution
        form_original = get_from_json(original, API_KEY)
        form_new = get_from_json(new, API_KEY)
        fe_defect = (form_original * -1) + form_new
        
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

def get_c_graph(structure):
    sites_list = structure.sites

    # The nodes: These are the sites features
    nodes = []
    for i, site in enumerate(sites_list):
        node_features = [
            i, 
            site.properties["bonds_broken"], 
            site.properties["original_an"], 
            site.properties["new_an"], 
            site.properties["an_change"], 
            site.properties["vacancy_defect"], 
            site.properties["substitution_defect"], 
            site.properties["site_fe"]
        ]

        # Node features syntax
        nodes.append(node_features)
         

    # The edges
    edges = [] # The sites in relation
    edge_features = [] # The distance between each site

    from_e = []
    to_e = []
    
    for i, site_i in enumerate(sites_list):
        for j, site_j  in enumerate(sites_list):
            # Edges 
            from_e.append(i)
            to_e.append(j)

            # Get distance between sites
            dist = site_i.distance(site_j)

            # Are the defects the same or different
            same_diff = int(site_i.properties["an_change"] == site_j.properties["an_change"])

            # What is the site_fe difference
            site_fe_diff = np.abs(site_i.properties["site_fe"] - site_j.properties["site_fe"])

            edge_features.append([dist,same_diff,site_fe_diff])
            
    edges.append(from_e)
    edges.append(to_e)

    # The global features
    the_ids = []
    the_ratios = []
    total_sites = len(sites_list)

    the_formula = structure.formula
    composition = Composition(the_formula)
    element_dict = composition.get_el_amt_dict()

    for symb, numb in element_dict.items():
        try:
            ids = Element(symb).Z - 1
        except ValueError:
            ids = 0
        the_ids.append(ids)
        ration = numb/total_sites
        the_ratios.append(ration)

    return nodes, edges, edge_features, the_ids, the_ratios
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader


from pymatgen.core import Structure, PeriodicSite, DummySpecie # , Composition, Element
# from pymatgen.core.periodic_table import Element as PMGElement
from pymatgen.analysis.local_env import MinimumDistanceNN, CrystalNN, VoronoiNN
# from mp_api.client import MPRester
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import json
import config

API_KEY = config.API_KEY


vnn = VoronoiNN(allow_pathological=True)

def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}

# Functions to get formation energy from Materials Project
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


"""def get_from_json(element, API_KEY):
    try:
        with open("./test.json", "r") as f:
            the_dict = json.load(f)
            to_return = the_dict[element]

    except:
        the_dict = {}
        with open("./test.json", "a") as f:
            to_return = get_formation(element, API_KEY)
            the_dict[element] = to_return

    return to_return"""

def get_from_json(element, API_KEY):
    the_dict = {
            "Mo":-10.9332, "S":-4.127, "W":-13.0106, "Se":-3.489,
            "B":-6.704, "N":-8.324, "Ga":-3.03, "In":-2.715,
            "P":-5.362, "V":-8.992, "O":-4.938, "C":-9.226
        }
    if element in the_dict:
        to_return = the_dict[element]
    else:
        to_return = get_formation(element, API_KEY)

    return to_return



def fe_site(original, new):
    if new == 0: # For vcancy
        fe_defect = get_from_json(original, API_KEY) * -1

    else: # For substitution
        form_original = get_from_json(original, API_KEY)
        form_new = get_from_json(new, API_KEY)
        fe_defect = (form_original * -1) + form_new

    return fe_defect

def get_ir(e):
    try:
        return float(e.average_ionic_radius)
    except Exception:
        try:
            return float(e.atomic_radius)
        except Exception:
            return 0.0

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
                defects_list.append(def_site)

                # Get atomic number change and defect type
                add_property = {
                    "original_Z":ref_specie.Z,
                    "new_Z": def_specie.Z,
                    "Z_change": def_specie.Z - ref_specie.Z,

                    "original_en": ref_specie.X,
                    "new_en": def_specie.X,
                    "en_change": def_specie.X - ref_specie.X,

                    "original_ir": get_ir(ref_specie),
                    "new_ir": get_ir(ref_specie),
                    "ir_change": get_ir(def_specie) - get_ir(ref_specie),

                    "original_ar": ref_specie.atomic_radius,
                    "new_ar": def_specie.atomic_radius,
                    "ar_change": def_specie.atomic_radius - ref_specie.atomic_radius,

                    "original_row": ref_specie.row,
                    "new_row": def_specie.row,
                    "row_change": def_specie.row - ref_specie.row,

                    "original_group": ref_specie.group,
                    "new_group": def_specie.group,
                    "group_change": def_specie.group - ref_specie.group,

                    "original_max_os": max(ref_specie.common_oxidation_states),
                    "new_max_os": max(def_specie.common_oxidation_states),

                    "original_ef": ref_specie.electron_affinity,
                    "new_ef": def_specie.electron_affinity,

                    "vacancy_defect": 0.0,
                    "substitution_defect": 1.0,
                    "bonds_broken": 0.0,
                    "site_fe": fe_site(ref_site.species_string,def_site.species_string),
                    "ref_idx": ref_index-1
                }

                voro_info = vnn.get_voronoi_polyhedra(reference_struct, add_property["ref_idx"])
                add_property["Voronoi_volume"] = sum(v["volume"] for v in voro_info.values())


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

            ref_specie = ref_site.specie

            # Get atomic number change and defect type
            add_property={
                "original_Z":ref_specie.Z,
                "new_Z": 0,
                "Z_change": 0 - ref_site.specie.Z,

                "original_en": ref_specie.X,
                "new_en": 0.0,
                "en_change": 0.0 - ref_specie.X,

                "original_ir": get_ir(ref_specie),
                "new_ir": 0.0,
                "ir_change": 0.0 - get_ir(ref_specie),

                "original_ar": ref_specie.atomic_radius,
                "new_ar": 0.0,
                "ar_change": 0.0 - ref_specie.atomic_radius,

                "original_row": ref_specie.row,
                "new_row": 0,
                "row_change": 0 - ref_specie.row,

                "original_group": ref_specie.group,
                "new_group": 0,
                "group_change": 0 - ref_specie.group,

                "original_max_os": max(ref_specie.common_oxidation_states),
                "new_max_os": 0,

                "original_ef": ref_specie.electron_affinity,
                "new_ef": 0.0,

                "vacancy_defect": 1.0,
                "substitution_defect": 0.0,
                "bonds_broken": mindnn.get_cn(reference_struct, reference_struct.sites.index(ref_site)),
                "site_fe": fe_site(ref_site.species_string,0),
                "ref_idx": ref_index-1
            }

            voro_info = vnn.get_voronoi_polyhedra(reference_struct, add_property["ref_idx"])
            add_property["Voronoi_volume"] = sum(v["volume"] for v in voro_info.values())


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

def get_nodes(defects_struct):
    defect_sites = defects_struct.sites

    nodes = []
    for site in defect_sites:
        site_features = [
            site.properties["Voronoi_volume"],
            site.properties["Z_change"],
            site.properties["ar_change"],
            site.properties["en_change"],
            site.properties["group_change"],
            site.properties["ir_change"],
            site.properties["new_Z"],
            site.properties["new_ar"],
            site.properties["new_ef"],
            site.properties["new_en"],
            site.properties["new_group"],
            site.properties["new_ir"],
            site.properties["new_max_os"],
            site.properties["new_row"],
            # site.properties["new_ve"],
            site.properties["original_Z"],
            site.properties["original_ar"],
            site.properties["original_ef"],
            site.properties["original_en"],
            site.properties["original_group"],
            site.properties["original_ir"],
            site.properties["original_max_os"],
            site.properties["original_row"],
            # site.properties["original_ve"],
            site.properties["row_change"],
            site.properties["bonds_broken"],
            site.properties["vacancy_defect"],
            site.properties["substitution_defect"],
            site.properties["site_fe"],
            # site.properties["ve_change"]
        ]
        nodes.append(site_features)

    return nodes


def get_edges_and_features(reference_structure, defects_structure):
    a_lat = float(reference_structure.lattice.a)

    from_edge = []
    to_edge = []
    edges = []
    edge_features = []

    cart_coords_ds = defects_structure.cart_coords

    for i, site_i in enumerate(defects_structure):
        for j, site_j in enumerate(defects_structure):
            if j > i :
                # pass
            # else:
                from_edge.append(i)
                to_edge.append(j)

                cart_i = cart_coords_ds[i]
                cart_j = cart_coords_ds[j]
                r_vec = cart_j - cart_i
                r_ij = float(np.linalg.norm(r_vec))

                # if r_ij > 12.0 or r_ij<1e-6:
                    # continue

                dist_angstrom = r_ij
                dist_norm = site_i.distance(site_j)
                dist_lattice_units = r_ij/a_lat

                # Formation energy interaction
                fe_site_i = site_i.properties["site_fe"]
                fe_site_j = site_j.properties["site_fe"]

                fe_product = fe_site_i * fe_site_j
                fe_sum = fe_site_i + fe_site_j
                fe_diff = abs(fe_site_i - fe_site_j)

                # Electrostatic interaction
                q_i = site_i.properties["new_max_os"]
                q_j = site_j.properties["new_max_os"]

                if site_i.properties["vacancy_defect"] == 1:
                    q_i = -q_i

                if site_j.properties["vacancy_defect"] == 1:
                    q_j = -q_j

                charge_product = q_i * q_j
                screened_coulomb = (q_i * q_j) / (r_ij) if r_ij > 0 else 0.0

                # Elastic size interaction
                ir_change_i = site_i.properties["ir_change"]
                ir_change_j = site_j.properties["ir_change"]

                ir_change_product = ir_change_i * ir_change_j
                elastic_size_interaction = (ir_change_i * ir_change_j) / (r_ij ** 3) if r_ij > 0 else 0.0

                # Angular factor
                if r_ij > 0:
                    cos_theta = r_vec[2]/ r_ij
                    angular_factor = 1.0-3.0 * cos_theta ** 2
                else:
                    angular_factor = 0.0

                # Defect interaction
                if site_i.properties["vacancy_defect"] == 1 and site_j.properties["substitution_defect"] == 1:
                    vac_sub = 1
                    vac_vac = 0
                    sub_sub = 0

                if site_i.properties["substitution_defect"] == 1 and site_j.properties["vacancy_defect"] == 1:
                    vac_sub = 1
                    vac_vac = 0
                    sub_sub = 0


                if site_i.properties["substitution_defect"] == 1 and site_j.properties["substitution_defect"] == 1:
                    vac_sub = 0
                    vac_vac = 0
                    sub_sub = 1

                if site_i.properties["vacancy_defect"] == 1 and site_j.properties["vacancy_defect"] == 1:
                    vac_sub = 0
                    vac_vac = 1
                    sub_sub = 0

                edge_features.append(
                    [
                        dist_angstrom, dist_norm, dist_lattice_units,
                        fe_product, fe_sum, fe_diff,
                        charge_product, screened_coulomb, ir_change_product,
                        elastic_size_interaction, angular_factor, 
                        vac_vac, sub_sub, vac_sub
                    ]
                )

    edges.append(from_edge)
    edges.append(to_edge)

    return edges, edge_features

def get_globals(pristine, defective_structure, defects_structure):
    p_n_species = len(pristine.composition.elements)
    d_n_species = len(defective_structure.composition.elements) 
    # Host composition vector: mean electronegativity, mean atomic radius, etc.
    p_elems = pristine.composition.elements
    d_elems = defective_structure.composition.elements

    p_ens = [e.X for e in p_elems]
    d_ens = [e.X for e in d_elems]

    p_ars = [e.atomic_radius for e in p_elems]
    d_ars = [e.atomic_radius for e in d_elems]

    p_host_mean_electronegativity = float(np.mean(p_ens)) if p_ens else 0.0
    p_host_electronegativity_spread = float(np.max(p_ens) - np.min(p_ens)) if p_ens else 0.0
    p_host_mean_atomic_radius = float(np.mean(p_ars)) if p_ars else 0.0

    d_host_mean_electronegativity = float(np.mean(d_ens)) if d_ens else 0.0
    d_host_electronegativity_spread = float(np.max(d_ens) - np.min(d_ens)) if d_ens else 0.0
    d_host_mean_atomic_radius = float(np.mean(d_ars)) if d_ars else 0.0

    # ---- Defect configuration summary ----
    n_defects = len(defects_structure)
    n_atoms_pristine = len(pristine)
    n_atoms_defective = len(defective_structure)
    defect_concentration = len(defects_structure) / len(pristine)

    vacs = 0
    subs = 0
    for def_site in defects_structure:
        if def_site.properties["vacancy_defect"] == 1:
            vacs += 1
        else:
            subs+=1

    n_vacancy = vacs
    n_substitution = subs

    global_list = [
        p_n_species, d_n_species, p_host_mean_electronegativity, p_host_electronegativity_spread, 
        p_host_mean_atomic_radius, d_host_mean_electronegativity, d_host_electronegativity_spread, 
        d_host_mean_atomic_radius, n_defects, n_atoms_pristine, n_atoms_defective, 
        defect_concentration, n_vacancy, n_substitution
    ]

    return global_list
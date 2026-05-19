# import pandas as pd
import numpy as np
from pymatgen.core import Structure, PeriodicSite, DummySpecie 

def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}


def get_defects_structure(defective_struct, reference_struct):
    # mindnn = MinimumDistanceNN()
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
                    "vacancy_defect": 0.0,
                    "substitution_defect": 1.0,
                }

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
                "vacancy_defect": 1.0,
                "substitution_defect": 0.0,

            }
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

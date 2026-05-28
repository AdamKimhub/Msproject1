# import pandas as pd
import numpy as np
from pymatgen.core import Structure, PeriodicSite, DummySpecie
import torch 


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


# Defects cloud to padded tensor
def cloud_to_tensor(defect_cloud, max_points):
    rows = []
    for a_site in defect_cloud:
        fc = a_site.frac_coords.tolist()

        required = [
            a_site.properties["original_Z"]/94,
            a_site.properties["new_Z"]/94,
            a_site.properties["substitution_defect"],
            a_site.properties["vacancy_defect"]
        ]

        combined = fc + required

        rows.append(combined)

    while len(rows) < max_points:
        rows.append([0.0]*7)

    return torch.tensor(rows, dtype=torch.float32)


def tensor_to_cloud(tensor_cloud, threshold):
    points = []
    for row in tensor_cloud:
        row = row.detach().cpu()
        if row.norm() < threshold:
            continue  # padding row
        frac_coords = row[:3].clamp(0.0, 1.0).numpy()

        Zp = int((row[3] * 94).round().clamp(0, 94).item())
        Zd = int((row[4] * 94).round().clamp(0, 94).item())

        sub_defect = row[5].item()
        vac_defect = row[6].item()

        if sub_defect:
            defect_type = "substitution"
        else:
            defect_type = "vacancy"

        points.append({
            "fractional_coords": frac_coords,
            "Z_pristine":  Zp,
            "Z_defective": Zd,
            "defect_type":  defect_type,
        })
    return points


def func_1(reference_structure):
    # Node features for each atom in pristine structure
    node_feats = []
    for site in reference_structure.sites:
        el = site.specie
        node_feats.append([
            el.Z / 94.0,
            el.X / 4.0 if el.X else 0.0,
            el.atomic_radius if el.atomic_radius else 0.0,
            el.row / 9.0,
            el.group / 18.0,
        ])
    node_features = torch.tensor(node_feats, dtype=torch.float32)


    # Build edge index from radius graph with cutoff of 5 Å
    src, dst = [], []
    coords = np.array([s.coords for s in reference_structure.sites])
    for i in range(len(reference_structure.sites)):
        for j in range(len(reference_structure.sites)):
            if i == j:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 5.0:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return{
        "node_features": node_features,
        "edge_index":    edge_index
    }


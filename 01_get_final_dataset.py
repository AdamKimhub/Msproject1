import pandas as pd
from pymatgen.core import Structure
import ast


# Standard df
init_structure_df = pd.read_csv(f"original_dataset/initial_structures.csv")
elements_df = pd.read_csv(f"original_dataset/elements.csv")

def main():
    materials = ["high_BN", "high_P", "high_InSe", "high_GaSe", "high_MoS2", "high_WSe2", "low_MoS2", "low_WSe2"]

    for i in materials:

        the_material = i.split('_')[1]
        
        reference_structure = get_reference_structures(i, the_material, save=True)
        ref_num_sites = len(reference_structure)

        defects_df, description_df = get_df(i)

        # Clearly represent the defects in the description_df
        description_df = description_df.apply(lambda row: dict_to_columns(row), axis= 1).fillna(0)

        # Add description to defects df
        merged_df = defects_df.merge(description_df, on="descriptor_id", how="left")

        # Replace the specific defect sites with type of defect sites
        merged_df = merged_df.apply(lambda row: get_to_strata(row, ref_num_sites), axis=1)
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if "vacant_" in col or "sub_" in col])

        # Get band gap and clean data 

        if "2" not in i: # high_BN, high_GaSe, high_InSe, high_P
            merged_df = merged_df.apply(remove_majmin, axis= 1)
            E1p, Evbmp = get_e_pristine(the_material)
            merged_df = merged_df.apply(lambda row: get_bgv(row, E1p, Evbmp), axis=1)

            merged_df = merged_df.drop(["defects", "descriptor_id", "homo_majority", "lumo_majority",
                                        "homo_lumo_gap_majority","E_1_majority", "homo_minority",
                                        "lumo_minority", "homo_lumo_gap_minority", "E_1_minority",
                                        "homo", "lumo", "description", "energy", "fermi_level",
                                        "E_1", "cell", "total_mag", "base"], axis=1)

        else:
            E1p, Evbmp = get_e_pristine(the_material)
            merged_df = merged_df.apply(lambda row: get_bgv(row, E1p, Evbmp),axis=1)
            if "high" in i: # high_MoS2, high_WSe2
                merged_df = merged_df.drop(["defects", "descriptor_id", "homo_lumo_gap",
                                        "homo", "lumo", "description", "energy",
                                        "fermi_level", "E_1", "cell", "base"], axis=1)

            elif "low" in i: # low_MoS2, low_WSe2
                merged_df = merged_df.drop(["defects", "descriptor_id", "homo_lumo_gap",
                                        "band_gap", "homo", "lumo", "description",
                                        "pbc", "energy", "fermi_level", "E_1", "cell",
                                        "base", "energy_per_atom"], axis=1)

        # Return the new df as csv
        new_csv_file = f"Final_Dataset/combined/{i}.csv"
        merged_df.to_csv(new_csv_file, index=False)

    all_df = [pd.read_csv(f"Final_Dataset/combined/{material}.csv") for material in materials]
    merged = pd.concat(all_df, ignore_index=True)

    # Get strata
    comb_df = get_strata(merged)

    comb_df.to_csv(f"Final_Dataset/combined/combined_data.csv", index=False)

# ===============================
# GENERATE REFERANCE STRUCTURE
# ===============================

def get_reference_structures(mat_dataset, base_material, save=False):
    # The unit cell structure
    unit_structure = Structure.from_file(f"original_dataset/{mat_dataset}/{base_material}.cif")

    # The cell matrix
    cell_matrix = init_structure_df.loc[init_structure_df["base"] == base_material, "cell_size"].iloc[0]
    cell_matrix = ast.literal_eval(cell_matrix)

    # Create the reference structure
    ref_structure = unit_structure.make_supercell(cell_matrix)

    if save:
        ref_structure.to(f"Final_Dataset/ref_cifs/{mat_dataset}.cif")

    return ref_structure

# ======================================
# PREPARE DEFECTS AND DESCRIPTOR FILES
# ======================================

def get_df(mat_dataset):

    # Load the data to df
    defects_df = pd.read_csv(f"original_dataset/{mat_dataset}/defects.csv")
    description_df = pd.read_csv(f"original_dataset/{mat_dataset}/descriptors.csv")

    # Prepare descrition_df
    description_df = description_df.rename(columns={"_id": "descriptor_id"})

    # Clearly specify the base for future stratification
    description_df["dataset_material"] = mat_dataset

    return defects_df, description_df

# ===================================
# PREPARE DEFECTS INFO
# ===================================

def string_to_dict(defects_string):

    defects_list = ast.literal_eval(defects_string)

    list_of_defects = []

    for i in range(len(defects_list)):
        if defects_list[i]['type'] == "vacancy":
            defect = f"vacant_{defects_list[i]['element']}"
            list_of_defects.append(defect)
        
        else:
            defect = f"sub_{defects_list[i]['from']}_{defects_list[i]['to']}"
            list_of_defects.append(defect)

    # Create a dictionary of defect_type: number_of_sites
    the_dict = {the_defect: list_of_defects.count(the_defect) for the_defect in list_of_defects}

    return the_dict

# =================================
# ADD THE DICTIONARIES AS COLUMNS
# =================================

def dict_to_columns(row):
    dict_defects = string_to_dict(row["defects"])

    for i,j in dict_defects.items():
        row[i] = j

    row.fillna(0.0, inplace=True)
    return row

# ==================================
# STRATA IN DATASET MATERIAL LEVEL
# ==================================

def get_to_strata(row, ref_num_sites):
    # Get the defects in the df
    all_columns = list(row.index)
    vacant_columns = [col for col in all_columns if "vacant" in col]
    sub_columns = [col for col in all_columns if "sub" in col]

    # Get defect:site pair
    vacant_dict = {i:row[i] for i in vacant_columns}
    vacants = sum(vacant_dict.values())

    sub_dict = {i:row[i] for i in sub_columns}
    subs = sum(sub_dict.values())

    # Get total defect sites
    defect_sites = vacants + subs


    # Get defect concentration
    defect_conc = round(defect_sites/ref_num_sites,5)

    # Other valuable columns
    row["vacancy_sites"] = vacants
    row["substitution_sites"] = subs
    row["defect_sites"] = defect_sites


    # The strata column will be in the form of material type_defect concentration
    row["to_strata"] = f"{row['base']}_{defect_conc}"
    return row

# ======================================
# HANDLE BAND GAP VALUE IN DIFF FILES
# ======================================

def remove_majmin(row):
    row["homo"] = (row["homo_majority"] + row["homo_minority"])/2
    row["lumo"] = (row["lumo_majority"] + row["lumo_minority"])/2
    row["E_1"] = (row["E_1_majority"] + row["E_1_minority"])/2

    return row

def get_e_pristine(the_material):
    E1_pristine = init_structure_df.loc[init_structure_df["base"] == the_material, "E_1"].iloc[0]
    Evbm_pristine = init_structure_df.loc[init_structure_df["base"] == the_material, "E_VBM"].iloc[0]

    return E1_pristine, Evbm_pristine

def get_bgv(row, E1_pristine, Evbm_pristine):
    new_norm_homo = row["homo"] - row["E_1"] - (Evbm_pristine - E1_pristine)
    new_norm_lumo = row["lumo"] - row["E_1"] - (Evbm_pristine - E1_pristine)

    row["band_gap_value"] = new_norm_lumo - new_norm_homo

    return row

# =============================
# STRATA FOR COMBINED DATA 
# =============================
def get_strata(merged_df):
    unique_values = pd.unique(merged_df["to_strata"])
    mapping = {value: i for i, value in enumerate(unique_values)}

    merged_df["strata"] = merged_df["to_strata"].map(mapping)
    merged_df = merged_df.drop(columns=["to_strata"])
    return merged_df

if __name__ == "__main__":
    main()
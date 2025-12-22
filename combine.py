import pandas as pd
from pymatgen.core import Structure

def main():
    original_data = "orignal_dataset"
    final_data = "final_dataset"
    
    materials = ["high_BN", "high_P", "high_InSe", "high_GaSe", "high_MoS2", "high_WSe2", "low_MoS2", "low_WSe2"]
    ref_sites_dict = {}

    for i in materials :
        # Get reference structure
        reference_structure = Structure.from_file(f"{final_data}/ref_cifs/{i}.cif")
        ref_sites_dict[i] = reference_structure.num_sites

        parts = i.split("_")
        the_material = parts[1]

        # Load the data to df
        defects_df = pd.read_csv(f"{original_data}/{i}/defects.csv")
        description_df = pd.read_csv(f"{original_data}/{i}/descriptors.csv")
        structure_df = pd.read_csv(f"{original_data}/initial_structures.csv")
        
        # Prepare the descriptor df
        # Change the column name of the descriptor id column
        description_df = description_df.rename(columns={"_id": "descriptor_id"})

        # Clearly represent the defects in the description_df
        description_df = description_df.apply(lambda row: string_to_columns(row), axis= 1).fillna(0)

        # Clearly specify the base for future stratification
        description_df["dataset_material"] = i

        # Add description to defects df
        merged_df = defects_df.merge(description_df, on="descriptor_id", how="left")

        # Modify the merged data
        # Target 
        if "2" not in i:
            merged_df = merged_df.apply(remove_majmin, axis= 1)
            merged_df = merged_df.apply(lambda row: get_bgv(row, structure_df, the_material), axis=1)

        # MoS2 and WSe2 only need to be normalized
        else:
            merged_df = merged_df.apply(lambda row: get_bgv(row,structure_df, the_material),axis=1)

        # Replace the specific defect sites with type of defect sites
        merged_df = merged_df.apply(lambda row: get_to_strata(row, ref_sites_dict), axis=1)
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if "vacant_" in col or "sub_" in col])

        # Remove the unrequired columns and add total mag where necessary
        if "2" not in i:
            merged_df = merged_df.drop(["defects", "descriptor_id", "homo_majority", "lumo_majority",
                                        "homo_lumo_gap_majority","E_1_majority", "homo_minority", 
                                        "lumo_minority", "homo_lumo_gap_minority", "E_1_minority",
                                        "homo", "lumo", "description", "energy", "fermi_level", 
                                        "E_1", "cell", "total_mag", "base"], axis=1)

        elif "2" in i and "high" in i:
            merged_df = merged_df.drop(["defects", "descriptor_id", "homo_lumo_gap", 
                                        "homo", "lumo", "description", "energy", 
                                        "fermi_level", "E_1", "cell", "base"], axis=1)

        elif "2" in i and "low" in i:
            merged_df = merged_df.drop(["defects", "descriptor_id", "homo_lumo_gap", 
                                        "band_gap", "homo", "lumo", "description", 
                                        "pbc", "energy", "fermi_level", "E_1", "cell",
                                        "base", "energy_per_atom"], axis=1)


        # Return the new df as csv
        new_csv_file = f"{final_data}/combined/{i}.csv"
        merged_df.to_csv(new_csv_file)

    
def string_to_sites(a_column):
    # Remove unwanted chars
    unwanted_chars = ['[',']']
    for i in unwanted_chars:
        a_column = a_column.replace(i,"")

    # Create a list of the different types of defects
    types = a_column.split("}")
    new_types = [j + "}" for j in types]

    # Remove the additional "{" at the end of the list
    del new_types[-1]

    # Remove the " ," before the "{"
    new_new_types = [types.lstrip(" ,") for types in new_types]

    # Defects clearly represented in 
    list_of_dicts = [eval(dict_string) for dict_string in new_new_types]

    list_of_defects = []
    for i in list_of_dicts:
        if i["type"] == "vacancy":
            defect = f'vacant_{i["element"]}'
            list_of_defects.append(defect)

        elif i["type"] == "substitution":
            defect = f'sub_{i["from"]}_{i["to"]}'
            list_of_defects.append(defect)

        else:
            list_of_defects.append("ubnormal")

    # Create a dictionary of defect_type: number_of_sites
    the_dict = {defect: list_of_defects.count(defect) for defect in list_of_defects}

    return the_dict

def string_to_columns(row):
    dict_defects = string_to_sites(row["defects"])

    for i,j in dict_defects.items():
        row[i] = j

    row.fillna(0.0, inplace=True)
    return row
    


def remove_majmin(row):
    row["homo"] = (row["homo_majority"] + row["homo_minority"])/2
    row["lumo"] = (row["lumo_majority"] + row["lumo_minority"])/2
    row["E_1"] = (row["E_1_majority"] + row["E_1_minority"])/2

    return row

def get_bgv(row, structure_df, base):
    E_1_pristine = structure_df.loc[structure_df["base"] == base, "E_1"].iloc[0]
    E_vbm_pristine = structure_df.loc[structure_df["base"] == base, "E_VBM"].iloc[0]

    new_norm_homo = row["homo"] - row["E_1"] - (E_vbm_pristine - E_1_pristine)
    new_norm_lumo = row["lumo"] - row["E_1"] - (E_vbm_pristine - E_1_pristine)

    row["band_gap_value"] = new_norm_lumo - new_norm_homo

    return row

def get_to_strata(row, ref_sites_dict):
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

    total_num_sites = ref_sites_dict[row["dataset_material"]]

    # Get defect concentration
    defect_conc = round(defect_sites/total_num_sites,5)

    # The strata column will be in the form of material type_defect concentration
    row["to_strata"] = f"{row['base']}_{defect_conc}"
    return row

def get_strata(merged_df):
    unique_values = pd.unique(merged_df["to_strata"])
    mapping = {value: i for i, value in enumerate(unique_values)}

    merged_df["strata"] = merged_df["to_strata"].map(mapping)
    merged_df = merged_df.drop(columns=["to_strata"])
    return merged_df

if __name__ == "__main__":
    main()

# Exploring the Band Gap Property of 2D Crystalline Materials With Point Defects Using a Machine Learning-based Model
This repository is set up to build a machine learning model similar to CGCNN to predict the band gap values of 2D crystalline materials with point defects(vacancy or substitutional).

The model primarily identifies patterns that exist between the type, number, and arrangement of defects in a crystalline 2D material and a material's band gap properties.

## Details of what each file does
### `combine.py`
This python file combines the `descriptor.csv` file with the `defects.csv` file while also doing some calculatons to extract vital features of each defective structure. 

Resultant features/atttributes are:
1. `_id`: Cif id of a crystalline strcture.
2. `energy`: Total potential energy of the crystal structure as reported by VASP(given in eV).
3. `fermi_level`: Fermi level of the crystalline structure given in eV
4. `total_mag`: Total magnetisation of the crystalline material
5. `base`: Host material of the crystalline material
6. `cell`: Supercell size
7. `vacancy_sites` and `substituiton_sites`: defect type identification
8. `dataset_material`: Type of dataset material(high density dataset+host material or low density + host material)[Essential for splitting the data]
9. `formation_energy`: defect formation energy
10. `formation_energy_per_site`: defect formation energy divided by the number of defect sites
11. `energy_per_atom`: Total potential of the system divide by the number of atoms(given in eV)
12. `E_1`: Energy of the first Kohn-sham orbital of the structure with defects
13. `norm_homo`: Normalized value of highest occcupied molecular orbital
14. `norm_lumo`: Normalized value of lowest unoccupoed molecular orbital.

After the `descriptor.csv` and the `defects.csv` files are combined, high density datasets are grouped together and the low density dataset is grouped together.

With this grouping, the low density dataset has a total of 3000 data points while the high density dataset has a total of 11,000 data points.

This means that there is an imabalance in the dataset. 


## Dataset Disclaimer
The dataset used in this repository is originally from **Pengru Huang**. If you wish to use this dataset, please **cite the original document** that generated it:

> Huang, Pengru, et al. "Unveiling the complex structure-property correlation of defects in 2D materials based on high throughput datasets." npj 2D Materials and Applications 7.1 (2023): 6.

I have modified the dataset to suit my needs, so the version provided here is **not identical to the original**.

To comply with proper attribution and avoid unintended reproduction of non-original documents, I have **not included the dataset in this repository**. If you need access to the original data, please refer to the original source linked above.

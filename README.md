# Exploring the Band Gap Property of 2D Crystalline Materials With Point Defects Using a Machine Learning-based Model
This repository is set up to build a machine learning model similar to CGCNN to predict the band gap values of 2D crystalline materials with point defects(vacancy or substitutional).

## The Data
The materials involved in this dataset are:
1. Boron nitride ($BN$)
2. Black phosphorus ($bP$)
3. Gallium selenide ($GaSe$)
4. Indium selenide ($InSe$)
5. Molybdenum disulfide ($MoS_2$)
6. Tungsten diselenide ($WSe_2$)

These materials may have the vacancy, substitution, or both defect sites at different concentrations. They can have a low concentration of defects (1, 2, or 3 defect sites) or a high concentration of defects(2.5, 5, 7.5, 10, and 12.5% defect sites).

The dataset provided provides a dataset of 5933 defect configurations of lowly concentrated structures of defects in $WSe_2$ and $MoS_2$ materials totaling 11866. This dataset prioritizes the vast possibilities of arrangement of defect sites in a material.

For the highly concentrated dataset, all the materials mentioned above have 100 structures with 2.5, 5, 7.5, 10, and 12.5% defect sites. This makes up 500 structures for each of the 6 materials. This dataset prioritizes the concentration of defect sites in a material.

The model learns to identify how the type of host material, the type of defect present, the concentration of the defects in the material and the arrangement of the said defects in the material affects the bandgp of the material and inturn predict the band gap of a defective material.

## Workflow
The model is built on all the datasets combined through combine.py in an attempt to make it diverse to work on any host material.

Afterwards, the structures are turned to graphs to build a GNN model as inspired by the Crystal Graph Convolution Neural Network.

## Dataset Disclaimer
The dataset used in this repository is originally from **Pengru Huang**. If you wish to use this dataset, please **cite the original document** that generated it:

> Huang, Pengru, et al. "Unveiling the complex structure-property correlation of defects in 2D materials based on high throughput datasets." npj 2D Materials and Applications 7.1 (2023): 6.

I have modified the dataset to suit my needs, so the version provided here is **not identical to the original**.

To comply with proper attribution and avoid unintended reproduction of non-original documents, I have **not included the dataset in this repository**. If you need access to the original data, please refer to the original source linked above.

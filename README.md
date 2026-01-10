# Exploring the Band Gap of 2D Crystalline Materials With Point Defects Using a Machine Learning-based Model
This repository is set up to build a GNN machine learning model that can accurately predict the band gap of 2D crystalline materials with point defects(vacancy or substitutional).

## The Data
The materials involved in this dataset are:
1. Boron nitride ($BN$)
2. Black phosphorus ($bP$)
3. Gallium selenide ($GaSe$)
4. Indium selenide ($InSe$)
5. Molybdenum disulfide ($MoS_2$)
6. Tungsten diselenide ($WSe_2$)

These materials may have the vacancy or substitution defects at different concentrations and arranged in a variety of ways. They can have a low concentration of defects (1, 2, or 3 defect sites) or a high concentration of defects(2.5%, 5%, 7.5%, 10%, or 12.5% defect sites).

The dataset with low concentration of defects has 5933 datapoints involving $WSe_2$ and $MoS_2$ materials totaling 11866. This dataset prioritizes the vast possibilities of arrangement of defect sites in a material.

For the highly concentrated dataset, all the materials mentioned above have 100 datapoints with 2.5%, 5%, 7.5%, 10%, and 12.5% defect sites. This makes up 500 datapoints for each of the 6 materials totaling 3000 datapoints. This dataset prioritizes the concentration of defect sites in a material.

The model learns to identify how the type of host material, the type of defect present, the concentration of the defects in the material and the arrangement of the said defects in the material affects the bandgp of the material and inturn predict the band gap of the defective material.

## Workflow
You can follow the way I moved from the original data (`original_dataset`)to the training of the model in the following way:

1. Run all cells in `01-original_to_clean.ipynb` to clean the original data and exctract the required features for this project but you don't need to since the final dataset(What this code prodces) is already provided in this repository. 
2. Run all cells in `cif_to_graph.ipynb`. This will be used to convert the crystal structures into graphs.(You will need to have a materials project API key to run a part of this code)
4. Execute all cells in `graph_to_mldata.ipynb` to convert all the structures into graphs and create train, validation, and test sets.
5. Execute all cells in `model_build.ipynb` to train and test the model. The model architecture is saved in `models.py`
6. Use `Demonstration.ipynb` to illustrate the model in action.

You can also jump straight to predicting the band gap with the model using `Demonstration.ipynb`. The model will be able to make accurate predictions of the band gap as long as:

1. The defects present in the material are either vacancies or substitutionals.
2. The material is a 2D crystalline material*
3. You have both the pristine and defective structure so as to be able to map out where the defects are located.

* _The model will make accurate predictions if tested on materials that it was trained on. The model might fail to make accurate predictions for a different material(from the ones used to train it)_

## Dataset Disclaimer
The dataset used in this repository is originally from **Pengru Huang**. If you wish to use this dataset, please **cite the original document** that generated it:

> Huang, Pengru, et al. "Unveiling the complex structure-property correlation of defects in 2D materials based on high throughput datasets." npj 2D Materials and Applications 7.1 (2023): 6.

I have modified the dataset to suit my needs, so the version provided here is **not identical to the original**.

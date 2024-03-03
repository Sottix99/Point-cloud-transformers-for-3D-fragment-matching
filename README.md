 # Pair Fragments
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This repository refers to my master's thesis in the Data Science graduate program at Sapienza University.

The code used as a reference and starting point of this work for Point Cloud Transformer (PCT) is : [here](https://github.com/qq456cvb/Point-Transformers) (Menghao implementation)

## Data:

<p align="center">
  <img src="figures/Couple0.gif" alt="animated" />
 Couple 0 of the Test set
</p>

* The overall data used in this work is the Rotated cuts dataset (RCD) that comes from the Broken3D archive: [here](https://deeplearninggate.roma1.infn.it/)
* The Train set used is : [here](https://drive.google.com/file/d/1k0u6Ycnizsu2SJv-FDkrXtIK6mOiOoca/view?usp=drive_link)
* The randomized Validation set used is: [here](https://drive.google.com/file/d/1UWc90jlblj_aks512WgtJRRJe9qyxEEO/view?usp=drive_link)
* The randomized Test set used is: [here](https://drive.google.com/file/d/17YF-sJryzKPkg8W-1FRWMt_62Y3cS-1o/view?usp=drive_link)


## Model:
The neural network developed for this thesis, as shown in the Figure, presents an architecture having two branches. In each of the two branches, there is the point cloud transformer encoder having shared weights. Compared to the original PCT encoder, modifications were made to allow compatibility with the data sizes used in this work. Each fragment has 7 features instead of the traditional three required by the first layer of the PCT encoder.
The input pairs are divided into two groups, one containing the first elements of each pair and the other the second. These tensors of fragments are processed in parallel in the two branches of the network through the pct encoder layers. The output of each branch represents the global features of the individual fragments input. The next step is to go and aggregate the two tensors produced to arrive at the global features of the pairs. Named $G_1$ and $G_2$, the global characteristics of the first and second elements of the pair, respectively, are aggregated through the use of two symmetrical functions, such as sum and multiplication, thus producing the global characteristics of the pairs ($G\_Tot$).

![My Imaged](figures/schema_2.png)
*Pair Model*

Then, $G\_{Tot}$ is input to the PCT's original classifier, which consists of three linear layers, where both relu and batch normalization are applied on the first two, interspersed with two dropout layers. 
In the output, the model generates predictions regarding the adjacency of the two elements forming the pair.  
## Results:
The following table shows the metrics for the three different runs performed, in the last column the link to download the weights of the trained model can be accessed.
| Number of Features        | Loss          | Accuracy  | F1 Score | AUC Score| Weights|
| ------------- |:-------------:| -----:|-----:|-----:|-----:|
| 3 | 0.628 | 0.650 | 0.649 | 0.698 | [Epoch 63](https://drive.google.com/file/d/1EUoxbtQ0-mnN-WUBiuLykIRP-mGSImaE/view?usp=sharing)|
| 6 | 0.621 | 0.655 | 0.655 | 0.709 | [Epoch 43](https://drive.google.com/file/d/13cdu3c3Adxyo_a0VtKkRIFbCrpav9bH0/view?usp=sharing)|
| 7 | 0.618 | 0.657 | 0.657 | 0.715 | [Epoch 116](https://drive.google.com/file/d/12wQAUwk6HGAq31u1YmNjTJ8JXqxPBJ_R/view?usp=drive_link)|

Another study was conducted to evaluate the effect of data augmentation on the model. The link to download the model weights without data augmentation is: [here (Epoch 98)](https://drive.google.com/file/d/1LikkbhCHqgWpocWq_R6fbTd2YsrnyRwb/view?usp=sharing).  The results of the comparison are shown in the following table:
|Train Data| Test Data | Loss          | Accuracy  | F1 Score | AUC Score|
| ------------- |:-------------:| -----:|-----:|-----:|-----:|
| Original Data | Original Data | 0.613 | 0.660 | 0.659 | 0.72  | 
| Original Data | Augmented Data| 0.625 | 0.648 | 0.644 | 0.707 | 
| Augmented Data| Original Data | 0.618 | 0.657 | 0.657 | 0.715 | 
| Augmented Data| Augmented Data| 0.617 | 0.659 | 0.659 | 0.715 | 




## Files:
* `Main.ipynb` notebook that contains the model, in which all the 7 features (x, y, z, nx, ny, nz, A) are used;
* `Main_3_features.ipynb` notebook that contains the model, in which only the  first 3 features (x, y, z) are used;
* `Main_6_features.ipynb` notebook that contains the model, in which only the  first 6 features (x, y, z, nx, ny, nz) are used;
* `Main_Not_Augmentation.ipynb` notebook that contains the model in which all 7 features (x, y, z, nx, ny, nz, A) are used, but data augmentation is not applied;
* `Inferences_on_modified_fragments.ipynb` notebook that contains the attempt of inferences when the fragments are modified;
* `Visualize_fragments.ipynb` notebook that contains the code to graphically represent the fragments, both original and modified;
* `environment.yml` the conda envirorment to run the models
* `miniconda3.yaml` the conda environment to print fragments in notebook `Visualize_fragments.ipynb`
  

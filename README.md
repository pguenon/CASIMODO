# CASIMODO
### _Conformation Analysis via Statistical Inference of MOlecular Dynamics Observables_
A script by <ins>Paul Guénon</ins>, Guillaume Stirnemann, Damien Laage and Olivier Rivoire*

## What is CASIMODO?

## Quick Start
In this section you will find all the information to easily start using CASIMODO in few minutes. If you want more details, please refer to the next sections. 

### Installing CASIMODO
To install CASIMODO, you will need a conda environment with a version of python 3.9 or higher.

You also need the following packages:
* numpy
* scipy
* sklearn
* matplotlib
* MDAnalysis
* dadapy

You will also need to download the *CASIMODO_utils/*  directory as well as the subimission file *submit_CASIMODO.sh* and to place them where you want to run CASIMODO.

You may also want to download the dictionnary file *dic_important_atoms_protein_nucleic_acids.txt* to inspire yourself when creating your own dictionnary file.

### Input files
As input files you need:
* A structure file with a format recognized by MDAnalysis (.pdb, .gro etc)
* A trajectory file with a format recognized by MDAnalysis (.trr, .xtc etc)
* A dictionnary file that contains the list of the important residue names in your simulation in the first column and the important atoms to study in each residue in the other columns. Optionnaly you may want to specify if a residue is an amino acid by adding the *@amino_acid* tag in the last column. Similarly you can specify if it is a nucleic acid with *@nucleic_acid_pyrimidine* and *@nucleic_acid_purine* tags.

### Running CASIMODO
Before running the script please make sure to indicate the following parameters in the *submit_CASIMODO.sh* file:
* *step_to_perform* sets the step you want to perform. Always start with the value **"all"**, when the first run is finished you can rerun other steps as described in the **Tuning Clustering** section.
* *struc_file* is the path to your structure file.
* *trj_file* is the path to your trajectory file.
* *dic_file* is the path to your dictionnary file that we described earlier.
* *output_directory* is the path to the directory where output files will be created. If the directory does not exist, it will be created.
* *time_zero* is the first time in your trajectory, in ps, to start the analysis. You can use this parameter to skip equilibration.
* *size_block* is the size of the time blocks, in ps, that will be used by CASIMODO during analysis. Indeed, during the analysis, distributions of geometric variables are plotted, and for checking convergence a block average is performed. If you don't want any block average please set this parameter to a very large value (longer than your simulation).
* *split_trajectory* this parameter tells the programm if you want to split the trajectory in conformations or not. Please be aware that splitting the trajectory leads to copying it multiple times and may produce large amount of data.

Once all mandatory parameters above are set you need to run the *submit_CASIMODO.sh* script on one (or more) CPU core. You may want to include this script in other scripts or to add a header to specify where to run the script. Please don't hesitate to do it, as long as you keep the script full integrity.

### Outputs
Here is a list of all ouputs produced by CASIMODO:
* *casimodo.log* this file contains is updated while the script is running and contains all the important information on the running.
* *important_atoms.txt* contains the list of the important atoms that were found in each of the selected residues.
* *selected_coordinates.txt* is the list of the coordinates that were found to be multimodal by CASIMODO. The first column is the name of the coordinates then the next ones are organized like this: **label0** **cutoff0** **label1** **cutoff1** **label2** . The smaller labels are assigned to the larger probabilities.
* *clusters_of_coordinates.txt* is the list of the clusters of coordinates that were found by the script, based on the variation of information, with the list of coordinates in each cluster.
* *resids_in_clusters.txt* is the list of the residues involved in each cluster of coordinates. This file is usefull for a quick visualization of the clusters but should'nt replace the detailed analysis of the coordinates in clusters.
* *conformations.txt* contains the list of the conformations that were found by CASIMODO for each cluster of coordinates. For each conformation, the probability of the conformation as well as the most probable state is printed.
* *discretizing_npy/* is a directory containing all the numpy arrays computed by CASIMODO during the discretizing step.
* *analysis_npy/* is a directory containing all the numpy arrays computed by CASIMODO during the analysis step.
* *coordinates_data/* contains the time evolution files for all the coordinates selected by CASIMODO.
* *coordinates_plots/* contains the distribution plots of all the coordinates selected by CASIMODO with discretizing cutoffs represented.
* *information_plots/* contains the plots for the entropy, the mutual information, the variation of information and the variation of information clustered for the selected coordinates.
* *conformations_clustering/* contains the plots for the clustering of states in each cluster of coordinates, as well as indexes for conformations in each cluster of coordinates and, if *split_trajectory*  is **True**, the splitted trajectories and a structure file with the same topology.

The most important outputs are probably ***clusters_of_coordinates.txt*** and ***conformations.txt*** because they contain information about the clusters of coordinates and about conformations, ax well as ***conformations_clustering/*** because it contains the splitted trajectories.

## How does CASIMODO work?
If you want to use CASIMODO at its full potential it's important that you know how it works. All the steps followed by the programm are described lower.

### Loading the trajectory
CASIMODO uses MDAnalysis to load the trajectory, therefore all file formats recognized by MDAnalysis will work with CASIMODO.

### Times filtering
The times of the trajectory are filtered keeping all times from *time_zero* and with a time step *delta_time*.

### Important atoms selections 
All residues listed in your dictionnary file are kept as important residues. All atoms from these residues listed in your dictionnary are kept as important atoms. 

Amino acids are listed, as well as nucleic acids (both pyrimidine and purine).

### Selection and discretization of the coordinates

#### Selection and discretization of distances 
The distances between each residues are computed in the following way.

Let's say we have a residue i and a resude j, in residue i there are n<sub>i</sub> important atoms and in residue j there are n<sub>j</sub> important atoms. This define n<sub>i</sub>*n<sub>j</sub> distances between i and j that are computed by CASIMODO. Then, CASIMODO only keeps the distance that get the lowest among all of them accross the simulation and defines it as d<sub>ij</sub>. 

If d<sub>ij</sub> gets lower than *cutoff_distance* during the trajectory then we try to discretize it, otherwise we do not select any distance between i and j.

To discretize d<sub>ij</sub> we start by computing the distribution of this distance along the trajectory using block averaging. Then we smooth it with a gaussian kernel density estimator. On the smoothed distribution we try to identify several modes using peaks and minima determination. For a mode to be selected as a real mode, it's integrated probability should be larger than *proba_cutoff*.

If d<sub>ij</sub> is multimodal then it is discretized based on the modes and this discretization is written inside the *selected_coordinates.txt* file, the plot of its distribution is saved *coordinates_plots/* and it's time evolution in *coordinates_data/*.

#### Selection and discretization of dihedral angles
If you have amino acids in your system, then CASIMODO Will try to discretize and select all the psi and phi angles in the same way as for distances.

If you have nucleic acids in your system, then CASIMODO Will try to discretize and select all the alpha, beta, gamma, delta, epsilon, zeta and chi angles in the same way as for distances.

#### Selection and discretization of other coordinates
You may give as inputs other coordinates to CASIMODO.

For that you need to fill the following parameters :
* *coordinates_to_add* is a list of paths to the time evolution files of the coordinates you want to add. In the time evolution files, the first column should be the time in ps and the second column be either the distance in A or the angle in °. 
* *type_coordinates_to_add* is a list of types for the coordinates to add (angle or distance).
* *residues_coordinates_to_add* is a list of residue numbers involved in each coordinates to add. If you want to put more than one residue per coordinate, just put an underscore in between each number. 

The coordinates to add are discretized and selected just like other variables.

#### Discretization of the conformational space
If we selected and discretized N variables, then each frame of the trajectory can now be expressed as a number of length N, where each digit X can take n<sub>X</sub> different values, n<sub>X</sub> being the multiplicity of the selected coordinate X. This discretization is saved in *discretizing_npy/discretized_array.npy*.

#### Information calculation
We then compute several information on our discretized trajectory:
* The single frequency for each value x on each variable X.
* The double frequencies for each values x,y on each couple of variables X, J.
* The entropy for each variable X is computed as H(X)=-sum<sub>x(</sub>p(x)logp(x))




## Advanced Parameters

## Tuning clustering
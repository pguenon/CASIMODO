# CASIMODO
### _Conformation Analysis via Statistical Inference of MOlecular Dynamics Observables_
A script by <ins>Paul Guénon</ins>, Guillaume Stirnemann, Damien Laage and Olivier Rivoire*

## What is CASIMODO?

## Quick Start
In this section we will describe how to easily start using CASIMODO in few minutes. If you want more details, please refer to the next sections. 

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
* A dictionnary file that contains the list of the residue names in your simulation in the first column and the important atoms to study in each residue in the other columns. Optionnaly you may want to specify if a residue is an amino acid by adding the *@amino_acid* tag in the last column. Similarly you can specify if it is a nucleic acid with *@nucleic_acid_pyrimidine* and *@nucleic_acid_purine* tags.

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
* *selected_coordinates* is the list of the coordinates that were found to be multimodal by CASIMODO. The first column is the name of the coordinates then the next ones are organized like this: **label_0** **cutoff_value_0** **label_1** **cutoff_value_1** **label_2** . The smaller labels are assigned to the larger probabilities.
* 

## How does CASIMODO work?

## Advanced Parameters

## Tuning clustering
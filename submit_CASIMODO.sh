#!/bin/bash

set -e  # Exit immediately if any command fails

##############################################
#               STEPS TO PERFORM             #
##############################################
# Choose which step to execute:
# Options:
#   all (default) — run the full pipeline
#   discretize_coordinates — select and discretize the CVs
#   cluster_coordinates — cluster CVs in communities based on Rajski's distance
#   get_conformations — define the states based on the communities of CVs and cluster them to get the conformations
#   plot_conformations_time — plot the conformations over time to see the transitions and compute the absolute Pearson's correlation coefficient between the several conformational trajectories

step_to_perform="all"

##############################################
#              REQUIRED INPUTS               #
##############################################
# Directory where all results will be saved
output_directory=results_CASIMODO_new_PDZ3_ligand

# topology file in a format readable by MDAnalysis
topol_file=Data_PDZ3/REST2_new_PDZ3_ligand_no_water_center_fit0_renumbered.gro 

# Trajectory file in a format readable by MDAnalysis, centered and fitted
trj_file=Data_PDZ3/REST2_new_PDZ3_ligand_no_water_center_fit.xtc 

# Dictionary file defining important residue types and their corresponding important atoms for the analysis.
dic_file=dic_important_atoms_protein_nucleic_acids.txt

##############################################
# ANALYSIS SETTINGS (Optional but important) #
##############################################
# Time in ps to start the analysis (to skip equilibration)
time_zero=0  
# Time (ps) between frames to consider. If smaller than actual trajectory temporal resolution, defaults to the actual timestep.
delta_time=0.0

##################################################### 
# CLUSTERING SETTINGS (To be finetuned iteritavely) #
#####################################################
#List of the possible methods for clustering coordinates and their associated parameters:
# 'hdbscan': (min_cluster_size min_samples cluster_selection_epsilon) - min_cluster_size is the minimum size of clusters, min_samples is a threshold parameter, the larger it is, the more points will be considered as noise, cluster_selection_epsilon is a distance threshold for merging clusters (default 0.0)
# 'yacare': (threshold_variable amount_of_noise keep_no_noise size_moving_square) - threshold_variable is a threshold variable, the smaller it is, the purer the clusters will be, amount_of_noise indicates the amount of data to recover from noise, keep_no_noise is a boolean indicating whether to consider noise for clustering (0 for noise, 1 for no noise), size_moving_square is the size of the moving square for the algorithm in percent
# 'ward': (threshold) - the smaller the threshold, the more clusters will be defined
# 'k-means': (n_clusters) - number of clusters to define

# Parameters for performing clustering of the coordinates
# Method for clustering coordinates (options: 'hdbscan', 'yacare')
method_clustering_coordinates="hdbscan"
# Parameters for clustering coordinates
parameters_clustering_coordinates=(10 10 0.5)

#Parameters for performing clustering of the conformations
# Method for clustering conformations (options: 'yacare', 'ward','k-means')
method_clustering_conformations="ward"
# Parameters for clustering conformations
parameters_clustering_conformations=(2.0)  
#Choose the community of coordinates to process. -1 for all communities, 0 for first community, 1 for second community, etc.
cluster_of_coordinates_to_process=-1 
# Whether to split the trajectory by conformations or not. 1 for True, 0 for False. 
split_trajectory=0 

###############################################
#       OUTPUT SETTINGS (Optional)         #
##############################################
# Extension for saved plots (e.g., png, pdf, svg)
extension_plots="png"  

# Resolution for saved plots in dpi
resolution_plots=200  

#Whether to save selected coordinates data
save_data=1  

# Whether to save all coordinates histogram plots, even the non discretized ones
save_all_plots=0  

##############################################
#           OPTIONAL COORDINATES             #
##############################################
# Additional coordinates to include in the analysis (e.g., RMSD, SASA, etc.)
coordinates_to_add=()

# Types corresponding to each additional coordinate (same order)
type_coordinates_to_add=()

# Residues to consider for additional coordinates
residues_coordinates_to_add=()

# Example usage:
# coordinates_to_add=( Data_files/angle1.dat Data_files/RMSD2.dat )
# type_coordinates_to_add=( angle distance )
# residues_coordinates_to_add=( 161_162 163_164_165_166 ) # Example for multiple residues per coordinate

##############################################
#  ADVANCED analysis SETTINGS (Optional)     #
##############################################
#Parameters for selecting distances to consider in the analysis: the distance between two atoms will be considered in the analysis if it is under cutoff_distance (in A) with a probability superior to proba_under_cutoff_distance
cutoff_distance=5
proba_under_cutoff_distance=0.01  

#Parameter for computing histograms: the bin size will be automatically determined based on the data so that there are at least n_points_per_bin points per bin, but it will not be smaller than min_bin_size_distances (in A) for distance-based coordinates and min_bin_size_angles (in °) for angle-based coordinates
n_points_per_bin=500  
min_bin_size_distances=0.1  
min_bin_size_angles=1.0  

#the bin size for KDE is the bin size for the histogram divided by smooth_factor. 
smooth_factor=10.0  

# Prominence for minima detection in discretization
prominence=0.025  # Prominence parameter for minima detection in discretization

# Cutoff number of points to use for computing histograms and KDE for discretization. If the number of points is superior to this cutoff, a random subset of points will be used for computing histograms and KDE to save time.
cutoff_npoints_discretization=100000  

# Minimal size of data to perform clustering. If the number of points in data to cluster is inferior to this cutoff, clustering will not be performed and all points will be considered as one cluster. 
minimal_size_to_cluster=10

#Cutoff for the number of states to consider when clustering the conformations. If the number of states is superior to this cutoff, only the most populated states will be clustered to save time in the analysis. 
cutoff_len_states=50000 

# Probability cutoff for conformations extraction. Only conformations with a probability superior to this cutoff will be extracted. 
cutoff_proba_conformations=0.0 

 

##############################################
#            MAIN EXECUTION BLOCK            #
##############################################
# Do not modify below unless you know what you're doing

python CASIMODO_utils/run_CASIMODO.py \
  --step_to_perform "${step_to_perform}" \
  --topol_file "${topol_file}" \
  --trj_file "${trj_file}" \
  --dic_file "${dic_file}" \
  --output_directory "${output_directory}" \
  --time_zero "${time_zero}" \
  --delta_time "${delta_time}" \
  --cutoff_distance "${cutoff_distance}" \
  --proba_under_cutoff_distance "${proba_under_cutoff_distance}" \
  --prominence "${prominence}" \
  --smooth_factor "${smooth_factor}" \
  --n_points_per_bin "${n_points_per_bin}" \
  --min_bin_size_distances "${min_bin_size_distances}" \
  --min_bin_size_angles "${min_bin_size_angles}" \
  --cutoff_npoints_discretization "${cutoff_npoints_discretization}" \
  --save_data ${save_data} \
  --save_all_plots ${save_all_plots} \
  --extension_plots "${extension_plots}" \
  --resolution_plots "${resolution_plots}" \
  --method_clustering_coordinates "${method_clustering_coordinates}" \
  --parameters_clustering_coordinates "${parameters_clustering_coordinates[@]}" \
  --method_clustering_conformations "${method_clustering_conformations}" \
  --parameters_clustering_conformations "${parameters_clustering_conformations[@]}" \
  --cluster_of_coordinates_to_process "${cluster_of_coordinates_to_process}" \
  --minimal_size_to_cluster "${minimal_size_to_cluster}" \
  --cutoff_len_states "${cutoff_len_states}" \
  --cutoff_proba_conformations "${cutoff_proba_conformations}" \
  --split_trajectory ${split_trajectory}\
  --coordinates_to_add "${coordinates_to_add[@]}" \
  --type_coordinates_to_add "${type_coordinates_to_add[@]}" \
  --residues_coordinates_to_add "${residues_coordinates_to_add[@]}" 


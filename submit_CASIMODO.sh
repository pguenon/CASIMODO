#!/bin/bash

set -e  # Exit immediately if any command fails

##############################################
#               STEPS TO PERFORM             #
##############################################
# Choose which step to execute:
# Options:
#   all (default) — run the full pipeline
#   discretize_coordinates — discretize the coordinates
#   cluster_coordinates — cluster based on mutual information
#   get_conformations — get the conformations from the clustered coordinates
#   plot_conformations_time — plot the conformations over time

step_to_perform="all"

##############################################
#              REQUIRED INPUTS               #
##############################################
# topology file (.gro or .pdb), centered and fitted
topol_file="struc.gro" # path to your structure file

# Trajectory file (.xtc or .trr), centered and fitted
trj_file="traj.xtc" # path to your trajectory file

# Dictionary file defining important atoms of residues
dic_file="dic_important_atoms_protein_nucleic_acids.txt" # path to your dictionary file

# Directory where all results will be saved
output_directory="results_CASIMODO" # path to your output directory

# Time in ps to start the analysis (to skip equilibration)
time_zero=0  # adjust as needed

# Whether to split the trajectory by conformations
# If True, the trajectory will be split into segments based on the identified conformations
split_trajectory=0 # 1 for True, 0 for False

############################################## 
#           CLUSTERING SETTINGS              #
##############################################
#Parameters for clustering of the coordinates

# Method for clustering coordinates (options: 'advanced_density_peaks', 'hdbscan', 'yacare')
method_clustering_coordinates="hdbscan"

# Parameters for clustering coordinates 
# for 'advanced_density_peaks': (Z_parameter halo_parameter)   halo_parameter is 0=False or 1=True ;
# for 'hdbscan': (min_cluster_size min_samples cluster_selection_epsilon)
# for 'yacare': (function_for_ratio threshold_variable amount_of_noise keep_no_noise size_moving_square) 
parameters_clustering_coordinates=(10 10 0)

#Parameters for clustering of the conformations
method_clustering_conformations="advanced_density_peaks"
# Parameters for clustering conformations
parameters_clustering_conformations=(1.0 0) # Example for yacare
#Choose the cluster of coordinates to process
cluster_of_coordinates_to_process=-1 # -1 for all clusters, 0 for first cluster, 1 for second cluster, etc.

minimal_size_to_cluster=10  # Minimal size to perform clustering
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
# coordinates_to_add=(Data_files/angle1.dat Data_files/RMSD2.dat)
# type_coordinates_to_add=(angle distance)
# residues_coordinates_to_add=( 161_162 163_164 ) # Example for multiple residues

##############################################
#        ADVANCED SETTINGS (Optional)        #
##############################################

# Time (ps) between frames to consider.
# If smaller than actual trajectory resolution, defaults to the actual timestep.
delta_time=1.0

#Parameters for contact analysis
# Distance threshold (Å) — two atoms are considered in contact
# if their distance is under cutoff_distance with a probability superior or equal to proba_under_cutoff_distance
cutoff_distance=5
proba_under_cutoff_distance=0.01  

# Ignore contacts between residues closer than this in the sequence
delta_residue=1

#Parameter for discretization
# Prominence for minima detection in discretization
prominence=0.025  # Prominence parameter for minima detection in discretization

smooth_factor=10.0  # Smoothing factor for discretization
min_bin_size_distances=0.1  # Minimum bin size for distance-based coordinates
min_bin_size_angles=1.0  # Minimum bin size for angle-based coordinates

n_points_per_bin=100  # Number of points per bin for density estimation

# Cutoff number of points to use for discretization
cutoff_npoints_discretization=100000  # Maximum number of points to use for discretization

# Probability cutoff for conformations extraction
cutoff_proba_conformations=0.001  # Probability cutoff for conformations extraction

#Cutoff number of states to consider in clustering states when searching for conformations
cutoff_len_states=10000  # Cutoff for the number of states to consider in clustering states

##############################################
#            MAIN EXECUTION BLOCK            #
##############################################
# Do not modify below unless you know what you're doing

python CASIMODO_utils/run_CASIMODO.py \
  --step_to_perform "${step_to_perform}" \
  -topol "${topol_file}" \
  -trj "${trj_file}" \
  -dic "${dic_file}" \
  --o_dir "${output_directory}" \
  --time_zero "${time_zero}" \
  --delta_time "${delta_time}" \
  --cutoff_distance "${cutoff_distance}" \
  --proba_under_cutoff_distance "${proba_under_cutoff_distance}" \
  --delta_resid "${delta_residue}" \
  --prominence "${prominence}" \
  --smooth_factor "${smooth_factor}" \
  --n_points_per_bin "${n_points_per_bin}" \
  --min_bin_size_distances "${min_bin_size_distances}" \
  --min_bin_size_angles "${min_bin_size_angles}" \
  --cutoff_npoints_discretization "${cutoff_npoints_discretization}" \
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


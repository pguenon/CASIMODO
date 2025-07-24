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

step_to_perform="all"

##############################################
#              REQUIRED INPUTS               #
##############################################
# Structure file (.gro or .pdb), centered and fitted
struc_file=" " # path to your structure file

# Trajectory file (.xtc or .trr), centered and fitted
trj_file=" " # path to your trajectory file

# Dictionary file defining important atoms of residues
dic_file=" " # path to your dictionary file

# Directory where all results will be saved
output_directory="results_CASIMODO" # path to your output directory

# Time in ps to start the analysis (to skip equilibration)
time_zero=0.

# Size in ps of each analysis block
size_block=100000000. # size of each analysis block in ps, if you want only one block, set it to a large value

# Whether to split the trajectory by conformations
# If True, the trajectory will be split into segments based on the identified conformations
split_trajectory=1 # 1 for True, 0 for False

############################################## 
#           CLUSTERING SETTINGS              #
##############################################
#Parameters for clustering of the coordinates

# Method for clustering coordinates (options: 'advanced_density_peaks', 'hdbscan', 'yacare')
method_clustering_coordinates="advanced_density_peaks"

# Parameters for clustering coordinates 
# for 'advanced_density_peaks': (Z_parameter halo_parameter)   halo_parameter is 0=False or 1=True ;
# for 'hdbscan': (min_cluster_size min_samples cluster_selection_epsilon)
# for 'yacare': (min_cluster_size min_samples)
parameters_clustering_coordinates=(1.65 1) 

#Parameters for clustering of the conformations
method_clustering_conformations="advanced_density_peaks"
# Parameters for clustering conformations
parameters_clustering_conformations=(0.7 0) 

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
# Probability cutoff for selection of a mode
mode_proba_cutoff=0.01

# Probability cutoff for conformations extraction
cutoff_proba_conformations=0.01  # Probability cutoff for conformations extraction


##############################################
#            MAIN EXECUTION BLOCK            #
##############################################
# Do not modify below unless you know what you're doing

python CASIMODO_utils/run_CASIMODO.py \
  --step_to_perform "${step_to_perform}" \
  -struc "${struc_file}" \
  -trj "${trj_file}" \
  -dic "${dic_file}" \
  --o_dir "${output_directory}" \
  --delta_time "${delta_time}" \
  --time_zero "${time_zero}" \
  --size_block "${size_block}" \
  --cutoff_distance "${cutoff_distance}" \
  --proba_under_cutoff_distance "${proba_under_cutoff_distance}" \
  --delta_resid "${delta_residue}" \
  --mode_proba_cutoff "${mode_proba_cutoff}" \
  --method_clustering_coordinates "${method_clustering_coordinates}" \
  --parameters_clustering_coordinates "${parameters_clustering_coordinates[@]}" \
  --method_clustering_conformations "${method_clustering_conformations}" \
  --parameters_clustering_conformations "${parameters_clustering_conformations[@]}" \
  --split_trajectory ${split_trajectory}\
  --cutoff_proba_conformations "${cutoff_proba_conformations}" \
  --coordinates_to_add "${coordinates_to_add[@]}" \
  --type_coordinates_to_add "${type_coordinates_to_add[@]}" \
  --residues_coordinates_to_add "${residues_coordinates_to_add[@]}" \
  


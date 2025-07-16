#!/bin/bash
set -e  # Exit immediately if any command fails

##############################################
#               STEPS TO PERFORM             #
##############################################
# Choose which step to execute:
# Options:
#   all (default) — run the full pipeline
#   discretize_coordinates — discretize the coordinates
#   get_frequencies — compute single and double frequencies
#   cluster_coordinates — cluster based on mutual information
#   get_conformations — get the conformations from the clustered coordinates

step_to_perform="all"

##############################################
#              REQUIRED INPUTS               #
##############################################
# Structure file (.gro or .pdb), centered and fitted
struc_file="Data_files/REMD_DHFR_WT_nowater_center_fit0.gro"

# Trajectory file (.xtc or .trr), centered and fitted
trj_file="Data_files/REMD_DHFR_WT_nowater_center_fit.xtc"

# Dictionary file defining important atoms of residues
dic_file="dic_important_atoms_protein_modified.txt"

# Directory where all results will be saved
output_directory="results_DHFR"

# Time in ps to start the analysis (to skip equilibration)
time_zero=150000

# Size in ps of each analysis block
size_block=50000

# Whether to split the trajectory by conformations
# If True, the trajectory will be split into segments based on the identified conformations
split_trajectory=True

##############################################
#        ADVANCED SETTINGS (Optional)        #
##############################################

# Time (ps) between frames to consider.
# If smaller than actual trajectory resolution, defaults to the actual timestep.
delta_time=1

#Parameters for contact analysis
# Distance threshold (Å) — two atoms are considered in contact
# if their distance is below this at least once
cutoff_distance=5
# Ignore contacts between residues closer than this in the sequence
delta_residue=3

#Parameter for discretization
# Minimum height difference (in %) between a local min and max
# to consider the local max as a significant peak
proba_cutoff=0.01

#Parameters for HDBSCAN clustering of the coordinates
# Minimum size of clusters for HDBSCAN
min_cluster_size_coordinates=5
# Minimum samples for HDBSCAN
min_samples_coordinates=40
# Epsilon for cluster selection in HDBSCAN
cluster_selection_epsilon_coordinates=0.0

#Parameters for HDBSCAN clustering of the conformations
# Minimum size of clusters for conformations extraction
min_cluster_size_conformations=100
# Minimum samples for conformations extraction
min_samples_conformations=100
# Epsilon for cluster selection in conformations extraction
cluster_selection_epsilon_conformations=0.0


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
# coordinates_to_add=(Data_files/RMSD.dat Data_files/SASA.dat)
# type_coordinates_to_add=(rmsd sasa)
# residues_coordinates_to_add=( 161_162 163_164 ) # Example for multiple residues

##############################################
#            MAIN EXECUTION BLOCK            #
##############################################
# Do not modify below unless you know what you're doing

if [ "${split_trajectory}" = "True" ]; then
  split_trajectory_flag="--split_trajectory"
else
  split_trajectory_flag="--no-split_trajectory"
fi

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
  --delta_resid "${delta_residue}" \
  --proba_cutoff "${proba_cutoff}" \
  --min_cluster_size_coordinates "${min_cluster_size_coordinates}" \
  --min_samples_coordinates "${min_samples_coordinates}" \
  --cluster_selection_epsilon_coordinates "${cluster_selection_epsilon_coordinates}"\
  --min_cluster_size_conformations "${min_cluster_size_conformations}" \
  --min_samples_conformations "${min_samples_conformations}" \
  --cluster_selection_epsilon_conformations "${cluster_selection_epsilon_conformations}" \
  ${split_trajectory_flag} \
  --coordinates_to_add "${coordinates_to_add[@]}" \
  --type_coordinates_to_add "${type_coordinates_to_add[@]}" \
  --residues_coordinates_to_add "${residues_coordinates_to_add[@]}" \
  


#!/bin/bash
set -e  # Exit immediately if any command fails

##############################################
#               STEPS TO PERFORM             #
##############################################
# Choose which step to execute:
# Options:
#   all (default) — run the full pipeline
#   discretize_conformations — discretize conformations
#   get_distances_between_coordinates — compute distances between coordinates
#   get_frequencies — compute frequency distributions
#   get_mutual_information — compute mutual information
#   get_entropy — compute entropy
#   clusterize_MI — cluster based on mutual information
#   extract_conformations — extract representative conformations
step_to_perform="all"

##############################################
#              REQUIRED INPUTS               #
##############################################
# Structure file (.gro or .pdb), centered and fitted
struc_file="Data_files/REMD_DHFR_WT_nowater_center_fit0.gro"

# Trajectory file (.xtc or .trr), centered and fitted
trj_file="Data_files/REMD_DHFR_WT_nowater_center_fit.xtc"

# Dictionary file defining terminal atoms of residues
dic_file="dic_terminal_atoms_protein_modified.txt"

# Directory where all results will be saved
output_directory="results_DHFR"

# Time in ps to start the analysis (to skip equilibration)
time_zero=150000

# Size in ps of each analysis block
size_block=50000

##############################################
#        ADVANCED SETTINGS (Optional)        #
##############################################
# Distance threshold (Å) — two atoms are considered in contact
# if their distance is below this at least once
cutoff_distances=5

# Ignore contacts between residues closer than this in the sequence
delta_residue=3

# Minimum height difference (in %) between a local min and max
# to consider the local max as a significant peak
height_cutoff=5

# Time (ps) between frames to consider.
# If smaller than actual trajectory resolution, defaults to the actual timestep.
delta_time=1

# Number of representative states to show per cluster
number_of_states_to_show=10

##############################################
#           OPTIONAL COORDINATES             #
##############################################
# Additional coordinates to include in the analysis (e.g., RMSD, SASA, etc.)
coordinates_to_add=()

# Types corresponding to each additional coordinate (same order)
type_coordinates_to_add=()

# Atom names used as barycenters for each additional coordinate (optional)
barycenter_coordinates_to_add=()

# Example usage:
# coordinates_to_add=(Data_files/RMSD.dat Data_files/SASA.dat)
# type_coordinates_to_add=(rmsd sasa)
# barycenter_coordinates_to_add=(45_CA 67_CB)

##############################################
#            MAIN EXECUTION BLOCK            #
##############################################
# Do not modify below unless you know what you're doing

python CASIMODO_utils/run_CASIMODO.py \
  -struc "${struc_file}" \
  -trj "${trj_file}" \
  -dic "${dic_file}" \
  --o_dir "${output_directory}" \
  --height_cutoff "${height_cutoff}" \
  --cutoff_distances "${cutoff_distances}" \
  --delta_resid "${delta_residue}" \
  --delta_time "${delta_time}" \
  --time_zero "${time_zero}" \
  --size_block "${size_block}" \
  --coordinates_to_add "${coordinates_to_add[@]}" \
  --type_coordinates_to_add "${type_coordinates_to_add[@]}" \
  --barycenter_coordinates_to_add "${barycenter_coordinates_to_add[@]}" \
  --step_to_perform "${step_to_perform}" \
  --number_of_states_to_show "${number_of_states_to_show}"

# Final message after completion
echo "Analysis complete. Results are saved in: ${output_directory}"

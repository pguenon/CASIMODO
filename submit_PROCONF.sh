#!/bin/bash

##################### STEP TO PERFORM ########################
step_to_perform=all # Options: all (default), discretize_conformations, get_distances_between_coordinates, get_frequencies, get_mutual_information, get_entropy, clusterize_MI, extract_conformations
##############################################################

###################### PARAMETERS TO MODIFY ##########################
struc_file=Data_files/REMD_DHFR_WT_nowater_center_fit0.gro
trj_file=Data_files/REMD_DHFR_WT_nowater_center_fit.xtc
dic_file=dic_terminal_atoms_protein_modified.txt
ouput_directory=results_DHFR # Directory where the results will be saved
time_zero=150000 # Time in ps to start the analysis
size_block=50000 # Size of the block to be analyzed in ps
######################################################################

######################## PARAMETERS TO MODIFY ONLY IF YOU KNOW WHAT YOU ARE DOING ##########################
# These parameters are set to default values, but you can modify them if needed
cutoff_distances=5 # Value in angstroms, to be considered as a contact, a distance should at least be once lower than this cutoff in the trajectory. 
delta_residue=3 # The contact between two residues that are closer than this delta in the sequence are not considered because they are arleady super close in the sequence, example: the first contact to be considered for residue 1 is 1-4 if delta_residue=3 
height_cutoff=1 # minimal difference of height, in percent of the maximum height, between a local minimum and a local maximum of a distribution to consider the local maximum as a peak
delta_time=1 # Time in ps between two frames to consider (if it is lower than the time step from the trajectory, then it takes instead the time step from the trajectory)
number_of_states_to_show=10 # Number of states to show for each cluster, after the clustering
############################################################################################################

########################### OPTIONAL PARAMETERS #################################
#optional parameters
coordinates_to_add=(Data_files/stacking_ligand_cofactor.dat) # Path to time evolution of coordinates to add to the analysis
# Example: coordinates_to_add=(Data_files/coordinate1.dat Data_files/coordinate2.dat)
type_coordinates_to_add=(angle) # Type of coordinates to add to the analysis
# Example: type_coordinates_to_add=(type1 type2)
barycenter_coordinates_to_add=(161_C4N) # Atom to consider as the barycenter of the coordinates to add
# Example: barycenter_coordinates_to_add=(1_CA 2_CB)
#################################################################################





####################################### DO NOT MODIFY ANYTHING BELOW THIS LINE ##############################
# Running the PROCONF script with the parameters defined above
python run_PROCONF.py -struc ${struc_file} -trj ${trj_file} -dic ${dic_file} --o_dir ${ouput_directory} --height_cutoff ${height_cutoff} --cutoff_distances ${cutoff_distances} --delta_resid ${delta_residue} --delta_time ${delta_time} --time_zero ${time_zero} --size_block ${size_block} --coordinates_to_add "${coordinates_to_add[@]}" --type_coordinates_to_add "${type_coordinates_to_add[@]}" --barycenter_coordinates_to_add "${barycenter_coordinates_to_add[@]}" --step_to_perform ${step_to_perform} --number_of_states_to_show ${number_of_states_to_show}
# The script will create a directory with the name ${ouput_directory} and will save the results there.

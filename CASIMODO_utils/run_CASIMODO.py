import os
import sys
import argparse

sys.path.append(os.getcwd()) # Add current directory to path for module imports

from CASIMODO_utils.functions_CASIMODO import *

#######################################
#         ARGUMENT PARSING            #
#######################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='CASIMODO - Conformational Information & Dynamics Entropy Reader')
    
    parser.add_argument('--step_to_perform', type=str, default='all', help='Step to perform in the pipeline')

    parser.add_argument('-struc', type=str, required=True, help='Path to GRO structure file')
    parser.add_argument('-trj', type=str, required=True, help='Path to trajector/y file')
    parser.add_argument('-dic', type=str, default='dic_important_atoms_protein.txt', help='Path to important atoms dictionary')
    parser.add_argument('--o_dir', type=str, default='results/', help='Output directory')
    

    parser.add_argument('--time_zero', type=int, default=150000, help='Time (ps) to start analysis')
    parser.add_argument('--size_block', type=int, default=50000, help='Size of block (ps) for analysis')
    parser.add_argument('-dt', '--delta_time', type=int, default=1, help='Time (ps) between frames to consider')

    parser.add_argument('--cutoff_distance', type=int, default=5, help='Distance cutoff (A) to define contact')
    parser.add_argument('--delta_resid', type=int, default=3, help='Residue separation threshold for contact filtering')
    parser.add_argument('--proba_cutoff', type=float, default=0.1, help='Probability cutoff for filtering dihedral regions')
    
    parser.add_argument('--Z_parameter_coordinates', type=float, default=3.0, help='Z parameter for clustering coordinates')
    parser.add_argument('--halo_parameter_coordinates', type=int, default=1, help='Halo parameter for clustering coordinates (0 or 1)')

    parser.add_argument('--Z_parameter_conformations', type=float, default=3.0, help='Z parameter for clustering conformations')
    parser.add_argument('--halo_parameter_conformations', type=int, default=0, help='Halo parameter for clustering conformations (0 or 1)')

    parser.add_argument('--cutoff_proba_conformations', type=float, default=0.001, help='Probability cutoff for conformations extraction')
    parser.add_argument('--split_trajectory', default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument('--coordinates_to_add', nargs='*', default=[], help='List of additional coordinate files')
    parser.add_argument('--type_coordinates_to_add', nargs='*', default=[], help='List of coordinate types (same order)')
    parser.add_argument('--residues_coordinates_to_add', nargs='*', default=[], help='List of residues to consider for additional coordinates (e.g., 161_162)')



    return parser.parse_args()


args = parse_arguments()

#######################################
#        VARIABLE INITIALIZATION      #
#######################################

step_to_perform = args.step_to_perform

strucfile = args.struc
trajfile = args.trj
dic = args.dic
output_dir = args.o_dir.rstrip('/') + '/'

time_zero = args.time_zero
size_block = args.size_block
delta_time = args.delta_time

cutoff_distance = args.cutoff_distance
delta_resid = args.delta_resid
proba_cutoff = args.proba_cutoff

Z_parameter_coordinates = args.Z_parameter_coordinates
halo_parameter_coordinates = args.halo_parameter_coordinates    

Z_parameter_conformations = args.Z_parameter_conformations
halo_parameter_conformations = args.halo_parameter_conformations

cutoff_proba_conformations = args.cutoff_proba_conformations
split_trajectory = args.split_trajectory

coordinates_to_add = args.coordinates_to_add
type_coordinates_to_add = args.type_coordinates_to_add
residues_coordinates_to_add = args.residues_coordinates_to_add




#######################################
#     CHECK INPUT FILE EXISTENCE      #
#######################################

for path in [strucfile, trajfile, dic]:
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.")
        exit(1)

#######################################
#      CREATE OUTPUT DIRECTORIES      #
#######################################

subdirs = [
    'coordinates_data',
    'coordinates_plots',
    'arrays_npy',
    'analysis',
    'information_plots',
    'frequencies',
    'conformations_clustering'
]

for subdir in subdirs:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

########################################
#           INITIATE LOGGING            #
########################################
initiate_logging(output_dir, step_to_perform)

########################################
#           PRINT HEADER               #
########################################
print_header()

#######################################
#         PRINT INPUTS                #
#######################################
print_inputs(
    output_dir, 
    step_to_perform, 
    strucfile, trajfile, dic,
    time_zero, delta_time, size_block,
    cutoff_distance, delta_resid, proba_cutoff,
    Z_parameter_coordinates, halo_parameter_coordinates,
    Z_parameter_conformations, halo_parameter_conformations,
    split_trajectory, cutoff_proba_conformations,
    coordinates_to_add, type_coordinates_to_add,residues_coordinates_to_add
)


#######################################
#       OPEN TRAJECTORY (if needed)   #
#######################################

if step_to_perform in ['all', 'discretize_coordinates','get_conformations']:
    u_traj = open_trajectory(strucfile, trajfile)

#######################################
#         TIME FILTERING              #
#######################################

if step_to_perform == 'all' :
    times, times_indices = filter_times_and_indices(u_traj, time_zero, delta_time, output_dir)

#######################################
#        GET TERMINAL ATOMS           #
#######################################

if step_to_perform in ['all', 'discretize_coordinates', 'get_conformations']:
    important_atoms_file = os.path.join(output_dir, 'important_atoms.txt')
    if os.path.exists(important_atoms_file):
        os.remove(important_atoms_file)
    
    important_atoms, selected_resids, selected_resnames, indices_aa, indices_na_pyrimidine, indices_na_purine = get_important_atoms_MDA(u_traj, dic)
    save_important_atoms(important_atoms, selected_resids, selected_resnames, output_dir)


#######################################
#     DISCRETIZE CONFORMATIONS        #
#######################################

if step_to_perform in ['all', 'discretize_coordinates']:
    selected_coordinates_file = os.path.join(output_dir, 'selected_coordinates.txt')
    if os.path.exists(selected_coordinates_file):
        os.remove(selected_coordinates_file)

    get_contacts(
        u_traj, important_atoms, selected_resids, time_zero, size_block,
        cutoff_distance, delta_resid, proba_cutoff, output_dir
    )
    if len(indices_aa)!=0 :
        get_dihedrals_protein(
            u_traj, indices_aa, time_zero, size_block,
            proba_cutoff, output_dir
        )

    if len(indices_na_pyrimidine) != 0 or len(indices_na_purine) != 0:
        get_dihedrals_nucleic_acids(
            u_traj, indices_na_pyrimidine, indices_na_purine, time_zero, size_block,
            proba_cutoff, output_dir
        )

    add_coordinates(
        coordinates_to_add, type_coordinates_to_add,
        time_zero, size_block,
        proba_cutoff,
        output_dir, 
    )

    get_discretized_array(output_dir)

    compute_information(output_dir)

#######################################
#           CLUSTERING STEP           #
#######################################

if step_to_perform in ['all', 'cluster_coordinates']:
    cluster_coordinates(
        output_dir, coordinates_to_add, residues_coordinates_to_add,
        Z_parameter_coordinates, halo_parameter_coordinates
        )
    
if step_to_perform in ['all', 'get_conformations']:
    get_conformations_from_clusters(
    output_dir,u_traj, 
    Z_parameter_conformations, halo_parameter_conformations,
    split_trajectory, cutoff_proba_conformations,strucfile,trajfile,selected_resids
    )

print_ending_message(output_dir, step_to_perform)

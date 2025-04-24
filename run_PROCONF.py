import numpy as np	
import argparse
import MDAnalysis as mda 
import os
from functions_PROCONF import *


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='get terminal atoms')  
    parser.add_argument('-struc', type=str, help='Path to gro file', required=True)
    parser.add_argument('-trj', type=str, help='Path to trajectory file', required=True)
    parser.add_argument('-non_regular_dic', type=str, default='dic_non_regular_residues.txt', help='Path to non regular residues dictionary file')
    parser.add_argument('-dic', type=str, default='dic_terminal_atoms_protein.txt',help='Path to terminal atoms dictionary file')
    parser.add_argument('--o_dir', type=str,default='results/', help='Path to output directory')
    parser.add_argument('--cutoff_distances', type=int,default=5,help='Cutoff index for the contacts')
    parser.add_argument('--delta_resid', type=int,default=3,help='Delta resid for the contacts')
    parser.add_argument('--time_zero', type=int,default=150000,help='Time zero in ps')
    parser.add_argument('--size_block', type=int,default=50000,help='Size of each block in ps')
    parser.add_argument('-dt','--delta_time', type=int,default=1,help='Delta time in ps')
    parser.add_argument('--height_cutoff', type=float, default=5, help='Height cutoff for the gaussians to be considered, in percent of the maximum height.')
    parser.add_argument('--coordinates_to_add', nargs='*', default=[], help='Path to file with coordinates to add')
    parser.add_argument('--type_coordinates_to_add', nargs='*', default=[], help='Type of coordinates to add')
    parser.add_argument('--barycenter_coordinates_to_add', nargs='*', default=[], help='Type of coordinates to add')
    parser.add_argument('--step_to_perform', type=str, default='all', help='Step to perform: get_terminal_atoms, get_contacts, get_dihedrals, add_coordinates, get_discretized_array, get_positions_baricenters, get_avg_distances_barycenters, get_frequencies, get_mutual_information, get_entropy, clusterize_MI')
    parser.add_argument('--number_of_states_to_show', type=int, default=10, help='Number of states to show after the clusterization step')
    args = parser.parse_args()
    return args
 
args=arg_parser()
strucfile=args.struc
trajfile=args.trj
dic=args.dic
output_dir=args.o_dir
if output_dir[-1]!='/':
    output_dir+='/'
cutoff_distances=args.cutoff_distances
delta_resid=args.delta_resid
time_zero=args.time_zero
size_block=args.size_block
delta_time=args.delta_time
height_cutoff=args.height_cutoff
non_regular_dic=args.non_regular_dic
coordinates_to_add=args.coordinates_to_add
type_coordinates_to_add=args.type_coordinates_to_add
barycenter_coordinates_to_add=args.barycenter_coordinates_to_add
step_to_perform=args.step_to_perform
number_of_states_to_show=args.number_of_states_to_show

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(output_dir+'coordinates_data'):
    os.mkdir(output_dir+'coordinates_data')
if not os.path.exists(output_dir+'coordinates_plots'):
    os.mkdir(output_dir+'coordinates_plots')
if not os.path.exists(output_dir+'Positions_npy'):
    os.mkdir(output_dir+'Positions_npy')
if not os.path.exists(output_dir+'analysis'):
    os.mkdir(output_dir+'analysis')
if not os.path.exists(output_dir+'MI_plots'):
    os.mkdir(output_dir+'MI_plots')
if not os.path.exists(output_dir+'frequencies'):
    os.mkdir(output_dir+'frequencies')

#################open traj####################################
if step_to_perform=='all' or step_to_perform=='discretize_conformations' or step_to_perform=='get_distances_between_coordinates' : 

    u_traj=open_trajectory(strucfile,trajfile)
##############################################################

#################get frames###########################
if step_to_perform=='all' :
    times, times_indices=filter_times_and_indices(u_traj, time_zero, delta_time,output_dir)
#################################################################

#################get terminal atoms###########################
if step_to_perform=='all' or step_to_perform=='discretize_conformations' or step_to_perform=='get_distances_between_coordinates' :
    if os.path.exists(f'{output_dir}terminal_atoms.txt'):
        os.system(f'rm {output_dir}terminal_atoms.txt')
    terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED,indices_aa=get_terminal_atoms_MDA(u_traj, dic)
    save_terminal_atoms(terminal_atoms, RESIDS_SELECTED, RESNAMES_SELECTED,output_dir)
#################################################################

#################discretize conformations############################
if step_to_perform=='all' or step_to_perform=='discretize_conformations':
    if os.path.exists(f'{output_dir}selected_coordinates.txt'):
        os.system(f'rm {output_dir}selected_coordinates.txt')
    get_contacts(u_traj,terminal_atoms,RESIDS_SELECTED,time_zero,size_block,delta_time,cutoff_distances,delta_resid,height_cutoff,indices_aa,output_dir)
    get_dihedrals(u_traj, indices_aa, time_zero, size_block, delta_time, cutoff_distances, delta_resid,height_cutoff,output_dir)
    add_coordinates(coordinates_to_add,type_coordinates_to_add,output_dir,time_zero,size_block,height_cutoff)
    get_discretized_array(output_dir)
#####################################################################

#################analysis of the conformations############################
if step_to_perform=='all' or step_to_perform=='get_distances_between_coordinates':
    get_positions_baricenters(u_traj,output_dir,RESIDS_SELECTED,indices_aa,terminal_atoms,coordinates_to_add,barycenter_coordinates_to_add)
    get_avg_distances_barycenters(output_dir)

if step_to_perform=='all' or step_to_perform=='get_frequencies':
    get_frequencies(output_dir)

if step_to_perform=='all' or step_to_perform=='get_mutual_information':
    get_mutual_information(output_dir)

if step_to_perform=='all' or step_to_perform=='get_entropy':
    get_entropy(output_dir)

if step_to_perform=='all' or step_to_perform=='clusterize_MI':
    if os.path.exists(f'{output_dir}Clusters_of_coordinate_from_MI.txt'):
        os.system(f'rm {output_dir}Clusters_of_coordinate_from_MI.txt')
    if os.path.exists(f'{output_dir}resids_in_cluster_from_MI.txt'):
        os.system(f'rm {output_dir}resids_in_cluster_from_MI.txt')
    clusterize_MI(output_dir,coordinates_to_add,barycenter_coordinates_to_add,step_to_perform,number_of_states_to_show)

if step_to_perform=='extract_conformations':
    cluster_states(output_dir)
#########################################################################



# CASIMODO
### _Conformational Analysis via Shared Information by MOlecular Dynamics Observables_

A script by _Paul Guénon_, Guillaume Stirnemann, Damien Laage, and Olivier Rivoire\*.

---

## What is CASIMODO?

**CASIMODO** is a Python-based tool designed to help automatically analyze conformational changes in molecular dynamics (MD) trajectories, especially in large and complex systems. It works by discretizing the conformational space and identifying the local variables that change state throughout the simulation.

The core idea behind CASIMODO is to provide a fast, lightweight, and user-friendly method for uncovering which parts of a system undergo structural changes and how. It is designed to run on a single CPU core and deliver results in a relatively short amount of time, making it accessible even on modest computing resources.

---

## Quick Start

This section walks you through the basic steps needed to get CASIMODO up and running in just a few minutes. If you're looking for more detail, feel free to read on into the later sections.

### Installation Requirements

To get started, you’ll need a Python environment (Python 3.9 or higher).

You’ll need the following Python packages installed:

- `numpy` < 2.4
- `scipy`
- `sklearn`
- `matplotlib`
- `MDAnalysis`
- `hdbscan`

Make sure to also download in your working directory:
- The `CASIMODO_utils/` directory,
- The job submission file `submit_CASIMODO.sh` that you will modify to create your own submission file,
- (Optional) The reference file `dic_important_atoms_protein_nucleic_acids.txt` to help you create your own dictionary.


### Input Files

To run CASIMODO, you need three key input files:

1. A **topology file** (e.g., `.pdb`, `.gro`) supported by MDAnalysis, that describes the molecular system.
2. A **centered trajectory file** (e.g., `.xtc`, `.trr`).  
  ⚠️ *CASIMODO does not handle periodic boundary conditions. You must preprocess and center your trajectory before analysis.*
3. A **dictionary file**, which lists:
    * Important residue names in the first column
    * Key atoms for each residue in subsequent columns

    You can also add tags to indicate whether a residue is:
    * An amino acid (`@amino_acid`)
    * A purine (`@nucleic_acid_purine`)
    * A pyrimidine (`@nucleic_acid_pyrimidine`)

This dictionary allows CASIMODO to focus on the relevant parts of your system for distance calculations and apply specific angle-based analyses when appropriate.

### Running CASIMODO

Before launching the script, open the file `submit_CASIMODO.sh` and fill in the following parameters:

* `step_to_perform`: Choose the step to execute. Begin with `"all"` for a full run. Later on, you can rerun specific steps (see Tuning Clustering).
* `topol_file`: Path to your topology file.
* `trj_file`: Path to your trajectory file.
* `dic_file`: Path to your dictionary file.
* `output_directory`: Where the output files will be saved. CASIMODO will create this directory if it doesn’t exist.

### Optional analysis settings 

* `time_zero`: The time (in ps) at which to begin analysis. Use this to skip the equilibration phase if needed.
* `delta_time`: The time (in ps) between frames to analyze. Setting this to a value larger than your trajectory’s native time step will speed up the analysis by sampling fewer frames.

You can modify or integrate this script into your own job submission pipeline, as long as its structure is preserved. CASIMODO should work smoothly in any environment where both Python and Bash are available.

### Tuning Clustering

Once the initial full run is complete, you may wish to explore better clustering results by adjusting certain parameters. CASIMODO allows you to rerun only the clustering steps to save time.

Use the `step_to_perform` variable in the submission script to specify the step:
* `"cluster_local_variables"`: Reruns the clustering of the geometric variables.
* `"get_conformations"`: Reruns the clustering of states and identification of conformations.

#### Parameters to adjust:
For both the clustering of the local variables and the clustering of the conformations you can choose a clustering method among `'hdbscan'`, `'yacare'`, `'ward'` and `'k-means'` and assign it to the parameters `method_clustering_local_variables`and `method_clustering_conformations`. The method can be different for clustering local variables and conformations. We advise using `'hdbscan'` for clustering local variables and `'ward'` for clustering conformations, but you can experiment with other methods as well.

You should then indicate the list of parameters you want to use for the clustering method you chose. To do so, you should enter the value of parameters one after the other with a withspace between successive parameters, and respecting the following orders.

The parameters to choose are the following ones, for more details about parameters please refer to litterature about the clustering methods:

**For hdbscan**:
- `min_cluster_size`: minimal size of the cluster you want. Integer superior or equal to 2. 
- `min_samples`: The higher it is, the purer the cluster. Usually choose it around the same value as `min_cluster_size`. Positive integer.
- `cluster_selection_epsilon`: Distance between clusters to be merged. Positive float.

**For Yacare**:
- `min_cluster_size`: minimal size of the cluster you want. Integer superior or equal to 2. 
- `threshold_variable`: The lower this variable, the purer the inital clusters.
- `amount_of_noise`: The higher it is, the more data will be removed from noise to be added to clusters.
- `keep_no_noise`: If `0` you have noise, if `1` you have no noise.
- `size_moving_square`: The size in percent of the total number of data points of the moving square used by yacare, by default use `2.0`. Positive float.

**For Ward**:
- `threshold`: The lower it is the purer the clusters. Positive float.

**For K-means**:
- `n_clusters`: The number of clusters you want. Positive integer.

You may need to experiment with these values to find a clustering result that best captures the behavior of your system.

When clustering conformations, two more parameters can be adjusted:
- `community_to_process`: The index of the community of variables to process. `-1` for all communities, `0` for first community, `1` for second community, etc.
- `split_trajectory`: If `1`, CASIMODO will save the trajectory segments corresponding to each conformation. This can be very useful for visual inspection of the conformations, but it can also take a lot of time and disk space. Keep it to `0`by default.

### Output Files
You can tune the outputs of CASIMODO by adjusting the following parameters:
- `extension_plots`: The file format for the plots (e.g., `png`, `pdf`). By default `png`.
- `resolution_plots`: The resolution of the plots in dots per inch (DPI). By default `200`.
- `save_data`: If `1`, data files containing the temporal evolutions of every selected local variable will be saved in the `local_variables_data/` directory. This can be useful for further analysis or custom plotting, but it can also take up disk space. Set to `0` by default to save space.
- `save_all_plots`: If `1`, the histograms for all analyzed variables will be saved in the `local_variables_plots/` directory. Even the ones that correspond to variables that are not selected. This can be useful for debugging. Set to `0` by default.


CASIMODO produces a number of output files and directories to help you interpret the results. Here are the key ones:

- `*.log`: Log files containing detailed information about the execution of each step.
- `important_atoms.txt`: List of important atoms identified from the dictionary.
- `selected_local_variables.txt`: List of all multimodal local variables and their discretization cutoffs.
- `communities_of_local_variables.txt`: Communities of local variables identified by clustering.
- `resids_in_communities_of_LVs.txt`: Residues associated with each community of local variables (mainly for quick inspection).
- `discretizing_npy/`: NumPy arrays from the discretization step.
- `analysis_npy/`: NumPy arrays from the analysis step.
- `local_variables_data/`: Time series of each selected local variable.
- `local_variables_plots/`: Distributions with cutoff lines of each selected local variable.
- `information_plots/`: Entropy and Rajski distance plots.
- `conformational_states_clustering/`: Conformational states clustering results, and if enabled, the split trajectory files and structure.

If you’re looking for the most critical outputs, focus on:
* `communities_of_local_variables.txt`
* `conformational_states_clustering/` (especially when `split_trajectory=1`)

---

## How Does CASIMODO Work?
Understanding CASIMODO’s internal workflow will help you make the most of it.

### 1. Trajectory Loading

CASIMODO uses **MDAnalysis** to handle structure and trajectory files, which supports most common formats.

### 2. Time Filtering

Only frames after `time_zero` are kept. Frames are sampled at an interval defined by `delta_time` if it is set to a value larger than the trajectory’s native time step.

### 3. Important Atom Selection

Important residues and atoms are selected based on your dictionary. If a residue is tagged as an amino acid or nucleic acid, CASIMODO will analyze its characteristic dihedral angles as well.

### 4. Coordinate Selection and Discretization

#### a. Distances

For each pair of residues, CASIMODO computes all pairwise distances between important atoms.

If a distance drops below `cutoff_distance` with a probability superior or equal to `proba_under_cutoff_distance`, the distance is considered for discretization.

Discretization involves:
* Building the distribution of the distance values across the trajectory
* Smoothing the distance distribution using a Gaussian kernel
* Detecting peaks and valleys
* Selecting modes corresponding to peaks of height superior to `prominence` times the maximum of the distribution.

Only multimodal distances are retained.

#### b. Dihedral Angles

For **Amino acids**: φ (phi) and ψ (psi)

For **Nucleic acids**: α, β, γ, δ, ε, ζ, χ

These are treated using the same selection and discretization process as distances.

#### c. User-Defined Coordinates
You can also input your own time-dependent variables:
- `local_variables_to_add`: List of file paths with coordinate values (first column: time in ps, second: value).  
  *For distances, use Ångströms; for angles, use degrees.*
- `type_local_variables_to_add`: Specify `"angle"` or `"distance"` for each.
- `residues_local_variables_to_add`: Residue indices involved (use underscores `_` to join multiple residues).

### 5. Discretization of Conformational Space
Each frame is represented as a list of discrete values (one per variable), forming a representation of the system based on the discretized variables. This is saved as `discretized_array.npy`.

### 6. Information-Theory Analysis

For each pair of selected variables, the following values are computed:

### Entropies

$H(X) = -\sum_x P(x) \log P(x)$

Measures the variability of a variable.

$H(X,Y) = -\sum_{x,y} P(x,y) \log P(x,y)$

Measures the joint variability of two variables.

### Rajski's Distance

$R^D(X,Y) = 1 - \frac{I(X,Y)}{H(X,Y)}$

With $I(X,Y)$ the mutual information between variables X and Y.

$I(X; Y) = \sum_{x,y} P(x, y) \log \left( \frac{P(x, y)}{P(x)P(y)} \right)$

The Rajski's distance is a measure of dissimilarity between two variables, with values between 0 and 1. A value of 0 indicates that the variables are identical, while a value of 1 indicates that they are completely independent.

## 7. Clustering the Coordinates

CASIMODO clusters variables into communities based on Rajski's distance using the algorithms described earlier.

Variables are grouped into communities representing independently changing subsystems.

## 8. Conformation Analysis

Once communities are defined, CASIMODO identifies conformational states for each of them:

1. Project trajectory into community subspace.
2. Define discrete states for each frame.
3. List all unique states observed.
4. Cluster the unique states.
5. Compute probability of each conformation.
6. Filter conformations with probability > `cutoff_proba_conformations`.
7. If `split_trajectory = 1`, extract trajectory segments for each conformation.

Outputs are saved in `conformations_clustering/`.

## Advanced Parameters

You may also want to adjust the following optional parameters in `submit_CASIMODO.sh` for more control: 

To tune distance selection:
- `cutoff_distance`: 	Distance threshold below which a coordinate is considered for analysis.
- `proba_under_cutoff_distance`: Probability cutoff for the distance threshold.

To tune histograms computation:
- `cutoff_npoints_discretization`: Maximal number of points to use for the histograms in the discretization step. If the number of points is superior to this value, a random subset of points will be used to compute the histograms. This can speed up the analysis for long trajectories, but it can also lead to less accurate histograms. Positive integer.
- `n_points_per_bin`: Minimal number of points per bin for the histograms used in the discretization step. 
- `min_bin_size_distabces`: Minimal size of the bins for the histograms of distances. In Ångströms.
- `min_bin_size_angles`: Minimal size of the bins for the histograms of angles. In degrees.
- `smooth_factor`: factor by which to divide the bin size to perform KDE smoothing of the histograms. The higher it is, the closer to the original histogram the smoothed histogram will be. Positive float.

To tune the selection of modes:
- `prominence`: The higher it is, the more prominent a peak must be to be selected as a mode. Positive float.

Optional parameter for clustering:
- `minimal_size_to_cluster`: If a matrix of distances has a size inferior to this value, it will not be clustered and each point will be considered as a cluster. 

Optional parameters for the clustering of conformations:
- `cutoff_n_configurations`: The maximum number of configurations to consider for the clustering of conformations. If the number of configurations is superior to this value, only the most probable configurations will be considered for the clustering. This can speed up the analysis for large systems, but it can also lead to missing some conformations. Positive integer.

- `cutoff_proba_conformations`: The probability cutoff for the conformations. Only conformations with a probability superior to this value will be kept. This can help focus on the most relevant conformations, but it can also lead to missing some important ones. Positive float between 0 and 1.

## License
License to come.

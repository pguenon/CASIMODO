# CASIMODO
### _Conformation Analysis via Statistical Inference of MOlecular Dynamics Observables_

A script by _Paul Guénon_, Guillaume Stirnemann, Damien Laage, and Olivier Rivoire\*.

---

## What is CASIMODO?

**CASIMODO** is a Python-based tool designed to help automatically analyze conformational changes in molecular dynamics (MD) trajectories, especially in large and complex systems. It works by discretizing the conformational space and identifying the geometric variables that evolve throughout the simulation.

The core idea behind CASIMODO is to provide a fast, lightweight, and user-friendly method for uncovering which parts of a system undergo structural changes—and how. It is designed to run on a single CPU core and deliver results in a relatively short amount of time, making it accessible even on modest computing resources.

This tool is especially valuable for early-career scientists who may find it challenging to analyze systems with many degrees of freedom. However, we believe experienced researchers will also find it useful due to its ease of use, speed, and ability to reduce the complexity of analysis without losing essential information.

---

## Quick Start

This section walks you through the basic steps needed to get CASIMODO up and running in just a few minutes. If you're looking for more detail, feel free to read on into the later sections.

### Installation Requirements

To get started, you’ll need a Python environment (Python 3.9 or higher).

You’ll need the following Python packages installed:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `MDAnalysis`
- `dadapy`

Make sure to also download in your working directory:
- The `CASIMODO_utils/` directory,
- The job submission file `submit_CASIMODO.sh`,
- (Optional) The reference file `dic_important_atoms_protein_nucleic_acids.txt` to help you create your own dictionary.


### Input Files

To run CASIMODO, you need three key input files:

1. A **structure file** (e.g., `.pdb`, `.gro`) supported by MDAnalysis, that describes the molecular system.
2. A **centered trajectory file** (e.g., `.xtc`, `.trr`).  
  ⚠️ *CASIMODO does not handle periodic boundary conditions. You must preprocess and center your trajectory before analysis.*
3. A **dictionary file**, which lists:
    * Important residue names in the first column
    * Key atoms for each residue in subsequent columns

    You can also add tags to indicate whether a residue is:
    * An amino acid (`@amino_acid`)
    * A purine (`@nucleic_acid_purine`)
    * A pyrimidine (`@nucleic_acid_pyrimidine`)

This dictionary allows CASIMODO to focus on the relevant parts of your system and apply specific angle-based analyses when appropriate.

### Running CASIMODO

Before launching the script, open the file `submit_CASIMODO.sh` and fill in the following parameters:

* `step_to_perform`: Choose the step to execute. Begin with `"all"` for a full run. Later on, you can rerun specific steps (see Tuning Clustering).
* `struc_file`: Path to your structure file.
* `trj_file`: Path to your trajectory file.
* `dic_file`: Path to your dictionary file.
* `output_directory`: Where the output files will be saved. CASIMODO will create this directory if it doesn’t exist.
* `time_zero`: The time (in ps) at which to begin analysis. Use this to skip the equilibration phase if needed.
* `size_block`: Size (in ps) of the time blocks used for convergence analysis and distribution calculation. If you want to skip block averaging, simply set this to a value larger than your total simulation time.
* `split_trajectory`: Set this to 1 if you want CASIMODO to split your trajectory into individual conformations.
    ⚠️ Note: This may generate large files depending on your trajectory size.

To run the script, use:

```bash
bash submit_CASIMODO.sh
```

You can modify or integrate this script into your own job submission pipeline, as long as its structure is preserved. CASIMODO should work smoothly in any environment where both Python and Bash are available.

### Tuning Clustering

Once the initial full run is complete, you may wish to explore better clustering results by adjusting certain parameters. CASIMODO allows you to rerun only the clustering steps to save time.

Use the `step_to_perform` variable in the submission script to specify the step:
* `"cluster_coordinates"`: Reruns the clustering of the geometric variables.
* `"get_conformations"`: Reruns the clustering of states and identification of conformations.

#### Parameters to adjust:

**For coordinate clustering**:
- `Z_parameter_coordinates`: Lower values result in more refined clusters but may assign more coordinates to noise.
- `halo_parameter_coordinates`: Set to `1` (recommended).

**For conformation clustering**:
- `Z_parameter_conformations`: A lower value increases purity of conformations, at the cost of more discarded states.
- `halo_parameter_conformations`: Set to `0` (recommended).

You may need to experiment with these values to find a clustering result that best captures the behavior of your system.

### Output Files

CASIMODO produces a number of output files and directories to help you interpret the results. Here are the key ones:

- `casimodo.log`: A log file containing the progress and key messages.
- `important_atoms.txt`: Lists important atoms identified from the dictionary.
- `selected_coordinates.txt`: Lists all multimodal coordinates and their discretization cutoffs.
- `clusters_of_coordinates.txt`: Coordinate clusters identified via VI.
- `resids_in_clusters.txt`: Residues associated with each cluster (mainly for quick inspection).
- `conformations.txt`: The conformations identified, their probabilities, and representative states.
- `discretizing_npy/`: NumPy arrays from the discretization step.
- `analysis_npy/`: NumPy arrays from the analysis step.
- `coordinates_data/`: Time series of each selected coordinate.
- `coordinates_plots/`: Distributions with cutoff lines of each selected coordinate.
- `information_plots/`: Entropy, mutual information, and variation of information visualizations.
- `conformations_clustering/`: States clustering results, and if enabled, the split trajectory files and structure.

If you’re looking for the most critical outputs, focus on:
* `clusters_of_coordinates.txt`
* `conformations.txt`
* `conformations_clustering/` (especially when `split_trajectory=1`)

---

## How Does CASIMODO Work?
Understanding CASIMODO’s internal workflow will help you make the most of it.

### 1. Trajectory Loading

CASIMODO uses **MDAnalysis** to handle structure and trajectory files, which supports most common formats.

### 2. Time Filtering

Only frames after `time_zero` are kept. Frames are sampled at an interval defined by `delta_time`.

### 3. Important Atom Selection

Important residues and atoms are selected based on your dictionary. If a residue is tagged as an amino acid or nucleic acid, CASIMODO will analyze its characteristic dihedral angles as well.

### 4. Coordinate Selection and Discretization

#### a. Distances

For each pair of residues, CASIMODO computes all pairwise distances between important atoms and retains the **minimum observed distance** over the trajectory, `d_ij`.

If `d_ij` drops below `cutoff_distance` at any point, the distance is considered for discretization.

Discretization involves:
* Smoothing the distance distribution using a Gaussian kernel
* Detecting peaks and valleys
* Selecting modes with integrated probabilities above `proba_cutoff`

Only multimodal distances are retained.

#### b. Dihedral Angles

For **Amino acids**: φ (phi) and ψ (psi)

For **Nucleic acids**: α, β, γ, δ, ε, ζ, χ

These are treated using the same selection and discretization process as distances.

#### c. User-Defined Coordinates
You can also input your own time-dependent coordinates:
- `coordinates_to_add`: List of file paths with coordinate values (first column: time in ps, second: value).  
  *For distances, use Ångströms; for angles, use degrees.*
- `type_coordinates_to_add`: Specify `"angle"` or `"distance"` for each.
- `residues_coordinates_to_add`: Residue indices involved (use underscores `_` to join multiple residues).

### 5. Discretization of Conformational Space
Each frame is represented as a list of discrete values (one per coordinate), forming a representation of the system based on the discretized coordinates. This is saved as `discretized_array.npy`.

### 6. Information-Theoretic Analysis

For each pair of selected coordinates, the following values are computed:

### Entropy

$H(X) = -\sum_x P(x) \log P(x)$

Measures the variability of a coordinate.

### Mutual Information

$I(X; Y) = \sum_{x,y} P(x, y) \log \left( \frac{P(x, y)}{P(x)P(y)} \right)$

Quantifies how much knowing one coordinate tells you about another.

### Variation of Information

$VI(X; Y) = H(X) + H(Y) - 2I(X; Y)$


A proper distance metric that is linked to mutual information and that forms the basis for clustering.

---

## 7. Clustering the Coordinates

CASIMODO clusters coordinates using **Advanced Density Peaks (ADP)**, implemented in `dadapy`. The clustering is based on the Variation of Information (VI) matrix between all pairs of coordinates.

Coordinates are grouped into clusters representing independently changing subsystems.

---

## 8. Conformation Analysis

Once coordinate clusters are defined, CASIMODO identifies conformations in each cluster:

1. Project trajectory into cluster subspace.
2. Define discrete states for each frame.
3. List all unique states observed.
4. Cluster the unique states using ADP.
5. Compute probability of each conformation.
6. Filter conformations with probability > `cutoff_proba_conformations`.
7. If `split_trajectory = 1`, extract trajectory segments for each conformation.

Output is saved in `conformations.txt` and (optionally) in `conformations_clustering/`.

---

## Advanced Parameters

These may be customized in `submit_CASIMODO.sh`:

- `delta_time`: Time step (ps) between frames.
- `cutoff_distance`: Minimum proximity (Å) for distances to be analyzed.
- `delta_residue`: Avoid intra-sequential residue distances (default: 1).
- `proba_cutoff`: Minimum integrated probability for a mode to be retained.
- `cutoff_proba_conformations`: Minimum probability for conformations.

---

## License

This software is distributed under the **MIT License**. See `LICENSE` for details.

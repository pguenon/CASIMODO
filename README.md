# CASIMODO
### _Conformation Analysis via Statistical Inference of MOlecular Dynamics Observables_

A script by _Paul Guénon_, Guillaume Stirnemann, Damien Laage, and Olivier Rivoire\*.

---

## What is CASIMODO?

CASIMODO is a tool designed to automatically analyze conformational changes in molecular dynamics (MD) simulations. It discretizes the conformational space of large biomolecular systems, revealing which geometric variables vary most over time and identifying the distinct conformations adopted throughout the simulation.

It is optimized to run efficiently on a single CPU core and aims to produce insightful results in a short time with minimal user intervention. While particularly helpful for beginners who may be overwhelmed by the number of degrees of freedom in large systems, CASIMODO is also valuable for experienced researchers who seek a fast and robust way to reduce the complexity of trajectory analysis—without losing important structural information.

CASIMODO can be run on any environment that supports **Python** and **Bash**.

---

## Quick Start

This section provides a brief walkthrough to get CASIMODO running in minutes. For more in-depth usage and customization, see the following sections.

### Installation Requirements

CASIMODO requires Python ≥3.9. A Conda environment is recommended.

Required Python packages:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `MDAnalysis`
- `dadapy`

Make sure to also download:
- The `CASIMODO_utils/` directory,
- The job submission file `submit_CASIMODO.sh`,
- (Optional) The reference file `dic_important_atoms_protein_nucleic_acids.txt` to help you create your own dictionary.

### Input Files

You will need the following:

- A **structure file** (e.g., `.pdb`, `.gro`) supported by MDAnalysis.
- A **centered trajectory file** (e.g., `.xtc`, `.trr`).  
  ⚠️ *CASIMODO does not handle periodic boundary conditions. You must preprocess and center your trajectory before analysis.*
- A **dictionary file** listing the important residues and atoms to analyze.  
  Optionally, use tags:
  - `@amino_acid`
  - `@nucleic_acid_pyrimidine`
  - `@nucleic_acid_purine`

### Running CASIMODO

In the `submit_CASIMODO.sh` script, you must define the following:

- `step_to_perform`: Initial run should be `"all"`.
- `struc_file`: Path to structure file.
- `trj_file`: Path to trajectory file.
- `dic_file`: Path to the dictionary file.
- `output_directory`: Output directory (created if missing).
- `time_zero`: Start time (ps) for analysis. Use to skip equilibration.
- `size_block`: Block size (ps) for time averaging. To disable block averaging, set it larger than the total simulation time.
- `split_trajectory`: `1` to split trajectory by conformations, else `0`.

Run the script on one or more cores:

```bash
bash submit_CASIMODO.sh
```

You may integrate this script into job submission systems as long as you preserve its integrity.

---

## Tuning Clustering

After an initial run, you can refine clustering via:

- `step_to_perform = "cluster_coordinates"`: Re-run coordinate clustering.
- `step_to_perform = "get_conformations"`: Re-run conformation identification.

Adjust the following parameters:

#### For coordinate clustering:
- `Z_parameter_coordinates`: Lower values increase purity but add noise.
- `halo_parameter_coordinates`: Set to `1` (recommended).

#### For conformation clustering:
- `Z_parameter_conformations`: Lower values increase purity but add noise.
- `halo_parameter_conformations`: Set to `0` (recommended).

Experiment with these to optimize clustering outcomes.

---

## Output Files

CASIMODO produces the following outputs:

- `casimodo.log`: Log of key runtime information.
- `important_atoms.txt`: Selected atoms for each important residue.
- `selected_coordinates.txt`: Multimodal coordinates with discretization.
- `clusters_of_coordinates.txt`: Coordinate clusters identified via VI.
- `resids_in_clusters.txt`: Residues involved in each coordinate cluster.
- `conformations.txt`: Probable conformations in each cluster.
- `discretizing_npy/`: Intermediate NumPy arrays from discretization.
- `analysis_npy/`: Arrays from entropy and clustering analysis.
- `coordinates_data/`: Time series of each selected coordinate.
- `coordinates_plots/`: Distributions with cutoff lines.
- `information_plots/`: Entropy, MI, VI plots.
- `conformations_clustering/`: Cluster plots, indices, and trajectory segments (if `split_trajectory=1`).

Key files include `clusters_of_coordinates.txt`, `conformations.txt`, and outputs in `conformations_clustering/`.

---

## How Does CASIMODO Work?

### 1. Trajectory Loading

CASIMODO uses **MDAnalysis** to load structure and trajectory files in supported formats.

### 2. Time Filtering

Only frames after `time_zero` are kept. Frames are sampled at an interval defined by `delta_time`.

### 3. Important Atom Selection

Residues listed in the dictionary are marked as important. CASIMODO selects the atoms listed for each residue, with special handling for amino acids and nucleic acids.

### 4. Coordinate Selection and Discretization

#### a. Distances

For each residue pair (i, j), the shortest distance among all pairs of important atoms is computed per frame. If this distance drops below `cutoff_distance` at any time, it is retained.

The distance is then:
- Histogrammed using block averages,
- Smoothed using a kernel density estimator,
- Discretized based on identified peaks and cutoffs.

Modes with integrated probability above `proba_cutoff` are retained.

#### b. Dihedral Angles

- **Amino acids**: φ (phi) and ψ (psi)
- **Nucleic acids**: α, β, γ, δ, ε, ζ, χ

These are discretized in the same manner as distances.

#### c. User-Defined Coordinates

- `coordinates_to_add`: List of file paths with coordinate values (first column: time in ps, second: value).  
  *For distances, use Ångströms; for angles, use degrees.*
- `type_coordinates_to_add`: List of `angle` or `distance`.
- `residues_coordinates_to_add`: Residue indices (use underscores `_` to join multiple residues).

---

## 6. Information-Theoretic Analysis

For each pair of selected coordinates:

### Entropy

\[
H(X) = -\sum_x P(x) \log P(x)
\]

Measures the variability of a coordinate.

### Mutual Information

\[
I(X; Y) = \sum_{x,y} P(x, y) \log \left( rac{P(x, y)}{P(x)P(y)} 
ight)
\]

Quantifies how much knowing one coordinate tells you about another.

### Variation of Information

\[
	ext{VI}(X, Y) = H(X) + H(Y) - 2I(X; Y)
\]

A proper distance metric that forms the basis for clustering.

---

## 7. Clustering the Coordinates

CASIMODO clusters coordinates using **Advanced Density Peaks (ADP)**, implemented in `dadapy`. The clustering is based on the Variation of Information (VI) matrix between all pairs of coordinates.

Each resulting cluster groups together coordinates that are functionally or dynamically related.

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

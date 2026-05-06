# Latent predictive pretraining improves a conservative neural force field on molecular dynamics trajectories

**Tanmay Sarkar Akash^1 and Siddhartha Das^1\***

**Affiliations:** ^1Department of Mechanical Engineering, University of Maryland, College Park, MD 20742

**Correspondence:** \*Email: sidd@umd.edu

**Unpublished research manuscript draft.** This manuscript is formatted in the approximate order recommended for Nature submissions: title, authors, affiliations, bold first paragraph, main text, references, figure legends, methods, data availability, code availability and declarations.

**Machine-learning force fields (MLFFs) can approach quantum-mechanical-level accuracy at a computational cost closer to classical force fields, but they remain constrained by the expense of labelled energy and force data. Here we test whether joint-embedding predictive architecture (JEPA) pretraining, adapted to atomistic trajectories, can improve a conservative neural molecular force field. Using the ethanol trajectory from the MD17 benchmark, we pretrained a SchNet-like continuous-filter encoder to predict latent embeddings of masked atomic environments at neighbouring time steps, then fine-tuned the same encoder to predict molecular energies and forces. In a 2,000/500/500 chronological train/validation/test split trained for 50 epochs, JEPA initialization reduced test force RMSE from 1.963 to 1.666 in native MD17 force units and reduced energy MAE from 4.266 to 1.058 in native MD17 energy units relative to a scratch baseline. These preliminary results suggest that latent predictive pretraining can transfer useful dynamical structure from unlabelled molecular trajectories into supervised neural force-field learning.**

## Main

Molecular dynamics (MD) requires a potential energy surface whose gradients produce physically meaningful forces. Classical force fields provide speed but can lack electronic accuracy, whereas direct ab initio MD is often too expensive for long trajectories or large systems. Machine-learned interatomic potentials have emerged as a compromise: they learn quantum-mechanical energy and force labels while retaining efficient evaluation in simulation. Gradient-domain force-field methods and the MD17 benchmark, continuous-filter neural networks and modern equivariant graph neural networks have established that molecular symmetries and conservative force construction are central to accuracy and data efficiency [1-6].

Despite these advances, most neural force-field training remains predominantly supervised. This is restrictive because high-quality quantum-mechanical energy and force labels require expensive electronic-structure calculations, whereas molecular geometries and trajectories from classical force fields, lower-level simulations or existing archives are easier to collect at scale. Self-supervised learning offers a possible route around this bottleneck by learning from coordinates before using scarce high-level labels. In the MD17 experiment below, the pretraining stage deliberately treats the available trajectory coordinates as unlabelled and ignores the accompanying energies and forces until fine-tuning. In vision, joint-embedding predictive architectures learn representations by predicting latent target embeddings from partial context rather than reconstructing raw pixels [6]. An analogous principle is appealing for molecules: atomic coordinates are continuous, noisy, symmetry-constrained and often locally ambiguous, but latent neighbourhood representations may capture stable dynamical regularities.

Here we implemented an atomistic JEPA pretraining workflow for a small-molecule neural force field. The model uses a compact SchNet-like continuous-filter encoder over atomic numbers and pairwise distances. The energy is invariant to rigid translations and rotations, and forces are obtained as negative coordinate gradients of the scalar energy, making the learned force field conservative and force predictions equivariant under rigid rotations. During pretraining, an online encoder receives a corrupted molecular frame in which a random subset of atoms is replaced by a learned mask token. An exponential-moving-average target encoder receives the uncorrupted frame at the next trajectory step. A predictor network is trained to match the target latent embeddings on masked atoms. The pretrained encoder is then fine-tuned using energy and force labels.

### Mathematical formulation

We represent an MD trajectory as a sequence of atomistic states

```math
x_t = (Z, R_t, E_t, F_t),
```

where \(Z \in \mathbb{N}^N\) are atomic numbers for \(N\) atoms, \(R_t \in \mathbb{R}^{N \times 3}\) are Cartesian coordinates, \(E_t \in \mathbb{R}\) is the reference potential energy and \(F_t \in \mathbb{R}^{N \times 3}\) are reference forces. The encoder \(f_\theta\) maps atomic numbers and coordinates to atomwise latent embeddings,

```math
H_t = f_\theta(Z, R_t) \in \mathbb{R}^{N \times d}.
```

For JEPA pretraining, we sample a binary atom mask \(m \in \{0,1\}^N\), where \(m_i=1\) indicates that atom \(i\) is hidden from the context branch. The online encoder receives a corrupted context frame \(\tilde{x}_t=(Z,\tilde{R}_t,m)\), implemented by replacing masked atom embeddings with a learned mask token, while the target encoder receives a clean future frame \(x_{t+\Delta}\). With online encoder \(f_\theta\), target encoder \(f_{\bar{\theta}}\), predictor \(q_\phi\) and stop-gradient operator \(\operatorname{sg}(\cdot)\), the pretraining loss is

```math
\mathcal{L}_{\mathrm{JEPA}}(\theta,\phi)
=
\frac{1}{\sum_i m_i}
\sum_{i=1}^{N}
m_i
\left\|
q_\phi\!\left(f_\theta(Z,\tilde{R}_t,m)_i\right)
-
\operatorname{sg}\!\left(f_{\bar{\theta}}(Z,R_{t+\Delta})_i\right)
\right\|_2^2 .
```

The target parameters are not optimized by backpropagation. Instead, they follow the online encoder through an exponential moving average,

```math
\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau)\theta,
```

with \(\tau=0.99\) in our experiments. After pretraining, the online encoder initializes the supervised force-field model.

The supervised model predicts a scalar potential energy as a sum of atomwise contributions,

```math
\hat{E}_\psi(Z,R)
=
b_E
+
\sum_{i=1}^{N}
g_\psi\!\left(f_\theta(Z,R)_i\right),
```

where \(g_\psi\) is an energy head and \(b_E\) is the train-set mean energy offset. Predicted forces are conservative by construction:

```math
\hat{F}_\psi(Z,R)
=
-\nabla_R \hat{E}_\psi(Z,R).
```

Fine-tuning minimizes a force-dominant supervised objective,

```math
\mathcal{L}_{\mathrm{sup}}
=
\lambda_E
\left\|
\hat{E}_\psi - E
\right\|_1
+
\lambda_F
\left\|
\hat{F}_\psi - F
\right\|_2^2 ,
```

with \(\lambda_E=0.05\) and \(\lambda_F=1.0\). Evaluation reports energy MAE, force MAE and force RMSE on the held-out test trajectory frames.

We evaluated the workflow on ethanol from MD17, a standard DFT molecular dynamics benchmark containing atomic numbers, coordinates, energies and forces. The experiment used 2,000 frames for training, 500 for validation and 500 for testing. Both scratch and JEPA-initialized force fields were trained for 50 epochs with identical supervised hyperparameters. The scratch baseline reached a best validation force RMSE of 1.596 and a final test force RMSE of 1.963. The JEPA-initialized model reached a best validation force RMSE of 1.322 and a final test force RMSE of 1.666. On the test set, JEPA pretraining improved force RMSE by 15.2%, force MAE by 12.6% and energy MAE by 75.2% relative to the scratch baseline (Table 1).

The representation pretraining itself converged rapidly. JEPA validation loss reached its minimum at epoch 9 (0.00119) and subsequently increased, indicating that the best self-supervised checkpoint may occur well before the end of a fixed 50-epoch pretraining schedule. This suggests that early stopping or a validation-selected JEPA checkpoint should be included in larger studies. Nevertheless, the final JEPA checkpoint still transferred effectively to supervised force-field training in this single-molecule experiment.

These results should be interpreted as a proof of concept rather than a benchmark claim. The encoder is intentionally lightweight and uses invariant scalar message passing rather than full tensor-equivariant interactions. The split is chronological and limited to one molecule, one random seed and one dataset size. The reported units are the native units stored in the downloaded MD17 `.npz` file and are not converted to the units used in all published rMD17 leaderboards. Even with these limitations, the controlled scratch-versus-pretrained comparison supports the central hypothesis: latent prediction over MD trajectories can provide useful initialization for conservative neural force fields.

The present experiment verifies one of two possible routes for JEPA-style force-field learning. In the verified route, both pretraining and fine-tuning use the same DFT trajectory distribution, but pretraining treats the coordinates as unlabelled and withholds the accompanying energies and forces. The improvement over a randomly initialized force field therefore demonstrates that a DFT-coordinate-only self-supervised stage can improve subsequent supervised DFT force-field learning in this ethanol setting. A second route is cross-fidelity pretraining, in which JEPA first learns from larger pools of classical, semiempirical or mixed-fidelity trajectories and is then fine-tuned only on high-level DFT labels. This route is attractive because lower-cost trajectories are easier to generate at scale, but it remains unverified in the present work and may help only when their coordinate distribution overlaps the target quantum-mechanical distribution.

Future work should test whether the effect persists across all rMD17 molecules, multiple random seeds, smaller labelled-data regimes and true tensor-equivariant backbones such as NequIP or MACE. The most important next experiment is label-efficiency: if JEPA pretraining provides larger gains when only tens to hundreds of labelled DFT frames are available, it would directly address the data bottleneck that limits practical ML force-field construction.

## Table 1 | Test performance after 50 supervised epochs

| Model | Energy MAE | Force MAE | Force RMSE | Drift proxy |
|---|---:|---:|---:|---:|
| Scratch | 4.266 | 1.458 | 1.963 | 1.458 |
| JEPA initialized | 1.058 | 1.274 | 1.666 | 1.274 |
| Relative improvement | 75.2% | 12.6% | 15.2% | 12.6% |

Metrics are reported in the native units of the downloaded MD17 ethanol `.npz` file. The drift proxy is the mean absolute force error and is included as a simple local surrogate for force-driven rollout error.

## References

1. Chmiela, S. et al. Machine learning of accurate energy-conserving molecular force fields. *Science Advances* **3**, e1603015 (2017). https://doi.org/10.1126/sciadv.1603015
2. Chmiela, S. et al. Towards exact molecular dynamics simulations with machine-learned force fields. *Nature Communications* **9**, 3887 (2018). https://doi.org/10.1038/s41467-018-06169-2
3. Schutt, K. T. et al. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems* **30** (2017). https://arxiv.org/abs/1706.08566
4. Batzner, S. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications* **13**, 2453 (2022). https://doi.org/10.1038/s41467-022-29939-5
5. Batatia, I. et al. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems* **35** (2022). https://arxiv.org/abs/2206.07697
6. Musaelian, A. et al. Learning local equivariant representations for large-scale atomistic dynamics. *Nature Communications* **14**, 579 (2023). https://doi.org/10.1038/s41467-023-36329-y
7. Assran, M. et al. Self-supervised learning from images with a joint-embedding predictive architecture. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (2023). https://arxiv.org/abs/2301.08243
8. Christensen, A. S. & von Lilienfeld, O. A. On the role of gradients for machine learning of molecular energies and forces. *arXiv:2007.09593* (2020). https://doi.org/10.48550/arXiv.2007.09593
9. Paszke, A. et al. PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems* **32** (2019).

## Figure legends

**Figure 1 | Force prediction scatter for scratch and JEPA-initialized models.** Predicted force components are plotted against DFT reference force components for the held-out MD17 ethanol test split. The diagonal line indicates perfect agreement. The JEPA-initialized model exhibits a lower force RMSE than the scratch baseline after the same number of supervised epochs. Source file: `results/force_scatter_50.png`.

**Figure 2 | Validation force RMSE during supervised fine-tuning.** Validation force RMSE is shown for scratch and JEPA-initialized force fields across 50 supervised epochs. JEPA initialization reaches a lower best validation force RMSE (1.322) than the scratch baseline (1.596). Source file: `results/loss_curves_50.png`.

## Methods

### Dataset

The ethanol trajectory was downloaded from the canonical sGDML/MD17 data endpoint using `scripts/download_md17.py`. MD17 was introduced for learning energy-conserving molecular force fields from *ab initio* molecular dynamics trajectories, and the revised MD17 literature motivates careful treatment of energy and force labels in molecular force-field benchmarks [1,8]. The validated file contained 555,092 frames and 9 atoms with atomic numbers `[6, 6, 8, 1, 1, 1, 1, 1, 1]`. Arrays were normalized to the internal key convention `z`, `R`, `E` and `F`. The experiment used the first 2,000 frames for training, the next 500 for validation and the next 500 for testing.

### Model

The encoder is a compact SchNet-like network with atom embeddings, Gaussian radial basis distance features, continuous-filter message passing and layer normalization. The force-field head predicts atomic energy contributions that are summed into a molecular energy. A train-set mean energy offset is stored as a non-trainable buffer. Forces are calculated by automatic differentiation as `F = -dE/dR`, ensuring a conservative force field.

### JEPA pretraining

The JEPA stage uses an online encoder, an EMA target encoder and a predictor MLP. For each molecular frame, a random subset of atoms is masked in the online branch. The target branch receives the uncorrupted frame at temporal offset `delta = 1`. The training loss is masked latent mean-squared error between predictor outputs and stop-gradient target embeddings. The default mask ratio is 0.35 and EMA decay is 0.99. The 50-epoch pretraining run wrote `checkpoints/jepa_pretrained_50.pt`.

### Supervised fine-tuning

Scratch and JEPA-initialized force fields were trained with the same supervised objective:

`L = 0.05 * MAE(E_pred, E_ref) + MSE(F_pred, F_ref)`.

Both models were trained for 50 epochs using AdamW. The scratch model was initialized randomly. The JEPA model loaded the pretrained encoder weights and initialized a fresh energy head.

### Evaluation

Evaluation was performed on the held-out 500-frame test set. Metrics were energy MAE, force MAE and force RMSE. A simple drift proxy was defined as mean absolute force error. This proxy is not a substitute for long-time MD rollout validation, but captures local force error relevant to short-step integration.

## Data availability

The MD17 ethanol dataset used in this study is publicly available from the sGDML/MD17 data distribution. The local validated dataset path for this run is `data/md17_ethanol.npz`. The metadata file is `data/md17_ethanol_metadata.json`.

## Code availability

All code used for this experiment is available at <https://github.com/blualg/jepa-mlff>. The implementation uses PyTorch [9].

## Acknowledgements

We thank the developers of MD17, sGDML, PyTorch and the broader machine-learning force-field community for datasets and software that make small-scale reproducible method testing possible.

## Competing interests

The authors declare no competing interests.

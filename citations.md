# Drift-Sense: Supporting References & Dataset Methodology

### Architecture & Methodology Citations

**1. Fully-Convolutional Siamese Networks for Object Tracking**
* **Citation:** Bertinetto, L., Valmadre, J., Henriques, J. F., Vedaldi, A., & Torr, P. H. (2016). *Fully-convolutional siamese networks for object tracking*. In ECCV.
* **Inherited Concept:** We adopted the twin-network architecture to extract uniform feature embeddings from both the reference template and the search image before comparison.

**2. SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks**
* **Citation:** Li, B., Wu, W., Wang, Q., Zhang, F., Xing, J., & Yan, J. (2018). *SiamRPN++*. In CVPR.
* **Inherited Concept:** We implemented depthwise cross-correlation in the feature space to compute similarity, enabling robust matching across extreme spatial drift.

**3. Objects as Points**
* **Citation:** Zhou, X., Wang, D., & Krähenbühl, P. (2019). *Objects as Points*. arXiv:1904.07850.
* **Inherited Concept:** We baked a fixed Log-Gaussian spatial penalty directly into the pre-activation logits, natively satisfying the "closest to center" tie-breaker rule without manual post-processing.

**4. Representation Learning with Contrastive Predictive Coding**
* **Citation:** Oord, A. v. d., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748.
* **Inherited Concept:** We applied the InfoNCE loss to treat the heatmap as a multiple-choice problem, actively suppressing identical periodic clones while maximizing the true target probability.

**5. CornerNet: Detecting Objects as Paired Keypoints**
* **Citation:** Law, H., & Deng, J. (2018). *CornerNet*. In ECCV.
* **Inherited Concept:** We utilized sub-pixel offset regression heads to precisely recover the exact nanometer coordinates lost during the network's spatial downsampling.

**6. Fast R-CNN**
* **Citation:** Girshick, R. (2015). *Fast R-CNN*. In ICCV.
* **Inherited Concept:** We employed Smooth L1 loss to optimize the sub-pixel offset, providing steady gradients for large errors while preventing gradient explosion on microscopic adjustments.

**7. Practical Poissonian-Gaussian Noise Modeling and Fitting**
* **Citation:** Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. (2008). *Practical Poissonian-Gaussian noise modeling...*. IEEE Transactions on Image Processing.
* **Inherited Concept:** We simulated the quantum-level arrival of electrons hitting the SEM sensor using a dose-scaled Poisson distribution to prevent model overfitting.

**8. Deep Learning for Automated SEM Image Defect Classification**
* **Citation:** Timofeeva, M., et al. (2020). *Deep Learning for Automated SEM Image Defect Classification*. Journal of Micro/Nanolithography, MEMS, and MOEMS.
* **Inherited Concept:** We mathematically modeled topological edge brightening to accurately simulate the increased secondary electron emission at sharp geometric corners in real SEM physics.

**9.  Geometric Augmentations & Manufacturing Variability**
* **Citation:** Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data.
* **Inherited Concept:** We utilized spatial affine augmentations (contact via rotation and local scaling jitter) to simulate standard physical manufacturing variability and prevent the network from overfitting to perfect procedural templates.

**10. SEM Raster Scanning Artifacts & Distortion**
* **Citation:** Sutton, M. A., et al. (2007). *Scanning electron microscopy for quantitative small and large deformation measurements*. Experimental Mechanics.
* **Inherited Concept:** Scanning Electron Microscopes acquire images sequentially, line-by-line. We implemented row-wise raster shear and random row jitter augmentations to mathematically simulate this physical beam-steering distortion.
---

### The Dataset Generation Journey

Developing the synthetic dataset required bridging the gap between ideal geometry and physical silicon. We began by rendering crisp, parameterized DRAM and FinFET layouts, but pure geometry failed to adequately challenge the neural network. To create a true "digital twin" of a scanning electron microscope, we introduced supersampled canvases, mat-specific pitch scaling for structural variance, and simulated mechanical raster shear. We then layered in rigorous physics models: modeling the Gaussian beam profile, increasing brightness at topological edges to mimic secondary electron escape, and applying localized charging artifacts. Finally, by capping the simulated mechanical stage drift to a realistic 800nm, we ensured that the periodic ambiguity remained a fair, solvable challenge rather than an impossible adversarial trap.

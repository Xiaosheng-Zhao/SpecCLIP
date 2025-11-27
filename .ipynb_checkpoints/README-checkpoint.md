# 🌌 SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-SpecCLIP-yellow)](https://huggingface.co/astroshawn/SpecCLIP/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.01939-b31b1b.svg)](https://arxiv.org/abs/2507.01939)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17717920.svg)](https://doi.org/10.5281/zenodo.17717920)

**A contrastive learning + domain-specific information preservation foundation model for cross-survey stellar spectroscopy**

> **SpecCLIP** aligns low-resolution LAMOST spectra with Gaia XP photometric spectra using CLIP-style contrastive learning while preserving domain-specific information.  
> It learns a *general-purpose spectral representation* that enables **parameter estimation**, **spectral translation**, and **similarity retrieval** across different spectroscopic surveys.

**Key Capabilities:**
- 🎯 Predict stellar parameters (T_eff, log g, [Fe/H], RV, chemical abundances) from LAMOST LRS and Gaia XP spectrum
- 🔄 Translate between different spectroscopic surveys (LAMOST ⟷ Gaia XP)
- 🔍 Retrieve similar stars from both LAMOST LRS spectra and Gaia XP spectra

**Down-Stream Tasks Overview**

<p align="center">
    <img width="900" src="./img/SpecCLIP_model_with_analysis_card.png"/>
</p>
---

## 🚀 Quick Start

### Installation

#### 1. Create and activate a new conda environment:
```bash
conda create -n astro-ai python=3.10
conda activate astro-ai
```

#### 2. Install PyTorch (CUDA 11.8 build)
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 3. Install key scientific stack from Conda
```bash
conda install numpy==2.0.1 scipy==1.15.3 pandas==2.3.3 mkl mkl-service -c defaults
```
#### 3. Install the remaining requirements via pip:
```bash
pip install -r requirements.txt
```
#### 4. Editable local install
```bash
pip install -e .
```

---

## 🧠 Pretrained Models

| Model                | Description                                          | Embedding Dim | Link                                                                       |
| -------------------- | ---------------------------------------------------- | ------------- | -------------------------------------------------------------------------- |
| `SpecCLIP-LRS`      | LAMOST LRS masked transformer    | 768           | [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/encoders/lrs_encoder.ckpt)      |
| `SpecCLIP-XP` | Gaia XP  masked transformer  | 768           | [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/encoders/xp_encoder.ckpt) |
| `SpecCLIP-XP-oae` | Gaia XP  ordinary auto-encoder  | 768           | [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/encoders/xp_encoder_mlp.ckpt) |
| `SpecCLIP-CLIP-pr` | Gaia XP  ⟷ LAMOST contrastive +pred+recon  | 768           | [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/specclip/specclip_model_predrecon_mlp.ckpt) |
| `SpecCLIP-CLIP-split` | Gaia XP  ⟷ LAMOST contrastive+(pred+recon-split)    | 768           | [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/specclip/specclip_model_split_mlp.ckpt) |

**Examples of training with your own data:**  
Refer to the `scripts/` directory. Before running any training script, create a `.env` file inside the `specclip/` folder and include:

```bash
SPECCLIP_ROOT="/path/to/your/specclip"
WANDB_ENTITY_NAME="your_wandb_entity"
```

---

## 🪄 Usage Examples

### Download model and data at once
```bash
!pip install -q huggingface_hub
from huggingface_hub import login
print("  1. Run in terminal: huggingface-cli login")
print("  2. Set the LOCAL_MODEL_DIR: LOCAL_MODEL_DIR='your_local_model_path')
!python download_and_setup.py 
```
### Parameter Prediction

Predict stellar parameters from any input spectrum:
```python
import json
from stellar_params_unified import UnifiedStellarParameterPredictor, get_default_config
from spectral_retrieval import SpectralRetriever
from IPython.display import display
import pandas as pd

# Configuration
with open('config_lrs.json', 'r') as f:
    lrs_config = json.load(f)

# Initialize for LAMOST LRS
predictor_lrs = UnifiedStellarParameterPredictor(survey_type='LAMOST_LRS', lrs_config=lrs_config)

# Predict All Parameters from LAMOST LRS
results_lrs_all = predictor_lrs.predict(
    './Foundation_LRS/test_data/sample1_matrix.fits',
    parameter_types=['all'],
    simple_header=False,
    display_format='row'
)

# Display with formatted style
display(predictor_lrs.display_results(results_lrs_all, style='formatted'))

```

### Spectral translation

Predict Gaia XP spectrum:
```python
import numpy as np
import matplotlib.pyplot as plt
import json
from spectral_retrieval import SpectralRetriever, get_default_model_paths
import sys
from predict_lrs_wclip_v0 import load_spectrum_data
from predict_xp_wclip_v0 import load_spectrum_data as load_spectrum_data_xp

# Display settings
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# Download est data only
!python download_and_setup.py --test-data-only

# Configuration
with open('config_retrieval.json', 'r') as f:
    config = json.load(f)

print(f"Test data path: {config.get('h5_data_path', 'Not configured')}")

# Initialize retriever
retriever = SpectralRetriever(**config)

# Build embedding database from test data
retriever.build_embedding_database(batch_size=1000, save_path='./test_embeddings.npz')

# Load pre-built embedding
retriever.load_embeddings('./test_embeddings.npz')

# Load the external spectra data
wavelength, flux = load_spectrum_data('./test_data/sample4_txt.csv')

# Predict corresponding Gaia XP spectrum
prediction_external = retriever.predict_cross_modal(
    query_spectrum=(wavelength, flux),
    query_type='lamost_spectra'
)

retriever.plot_cross_modal_prediction(
    prediction_external,
    save_path='./plots/external_lamost_to_gaia_prediction.png'
)

```

### Spectral Similarity Search

Find the 4 most similar stars from Gaia XP catalog:
```python
# Load external LAMOST spectrum
wavelength, flux = load_spectrum_data('./test_data/sample1_matrix.fits')

# Find similar Gaia XP spectra
results_external_cross = retriever.find_similar_spectra(
    query_spectrum=(wavelength, flux),
    query_type='lamost_spectra',
    search_type='cross_modal',
    top_k=4
)

print("Cross-modal retrieval: LAMOST → Gaia XP")
print(f"Top {len(results_external_cross['top_spectra'])} Gaia matches found")
print(f"Scores: {results_external_cross['top_scores']}")

retriever.plot_retrieval_results(
    results_external_cross,
    save_path='./plots/external_lamost_to_gaia_cross.png'
)
```

---

## 📊 Citation

If you use **SpecCLIP** in your research, please cite both the paper and the software:

```bibtex
@article{Zhao2025SpecCLIP,
  author        = {Zhao, Xiaosheng and others},
  title         = {SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars},
  journal       = {arXiv e-prints},
  year          = {2025},
  eprint        = {2507.01939},
  doi           = {10.48550/arXiv.250701939},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM}
}

@software{Zhao2025SpecCLIPSoftware,
  author       = {Zhao, Xiaosheng and others},
  title        = {SpecCLIP: A Foundation Model for Stellar Spectroscopy},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.17717920},
  url          = {https://doi.org/10.xxxx/zenodo.17717920}
}
```
---

## 🪐 Acknowledgments

We acknowledge the open-source [AstroCLIP](https://github.com/PolymathicAI/AstroCLIP) code base, upon which this project is largely built. 

---

## 📬 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/Xiaosheng-Zhao/SpecCLIP/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Xiaosheng-Zhao/SpecCLIP/discussions)
- **Email:** [xzhao113@jh.edu]

**Star ⭐ this repo if you find it useful!**

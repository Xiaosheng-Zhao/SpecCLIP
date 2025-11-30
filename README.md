# 🌌 SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-SpecCLIP-yellow)](https://huggingface.co/astroshawn/SpecCLIP/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.01939-b31b1b.svg)](https://arxiv.org/abs/2507.01939)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A contrastive learning + domain-specific information preservation foundation model for cross-survey stellar spectroscopy**

> **SpecCLIP** aligns low-resolution LAMOST spectra with Gaia XP photometric spectra using CLIP-style contrastive learning while preserving domain-specific information.  
> It learns a *general-purpose spectral representation* that enables **parameter estimation**, **spectral translation**, and **similarity retrieval** across different spectroscopic surveys.

**Key Capabilities:**
- Predict stellar parameters (T_eff, log g, [Fe/H], RV, varied chemical abundances, extinction, etc.) from LAMOST LRS and Gaia XP spectrum
- Translate between different spectroscopic surveys (LAMOST ⟷ Gaia XP)
- Retrieve similar stars from both LAMOST LRS spectra and Gaia XP spectra

**Model Architecture and Down-Stream Tasks Overview**

<p align="center">
    <img width="1000" src="./img/SpecCLIP_model_with_analysis_card.png"/>
</p>

---

## Quick Start

### Installation

#### 1. Create and activate a new conda environment:
```bash
conda create -n specclip-ai python=3.10
conda activate specclip-ai
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

## Pretrained Models

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

## Usage Examples
See the full demo here: [Full Demo](./downstream-with-demo/demo_download_v0.ipynb)
### Download model and data at once
```bash
!pip install -q huggingface_hub
from huggingface_hub import login

# Run in terminal to login huggingface: huggingface-cli login

# Set the path of LOCAL_MODEL_DIR in download_and_setup.py and download model/test data
!python download_and_setup.py 
```
### Parameter Prediction

Predict stellar parameters from any input spectrum:
```python
import json
from stellar_params_unified import UnifiedStellarParameterPredictor
from IPython.display import display

# Configuration
with open('config_lrs.json', 'r') as f:
    lrs_config = json.load(f)
predictor_lrs = UnifiedStellarParameterPredictor(survey_type='LAMOST_LRS', lrs_config=lrs_config)

# Predict All Parameters from LAMOST LRS
results_lrs_all = predictor_lrs.predict(
    './test_data/lrs/sample1_matrix.fits',
    parameter_types=['all'],
    simple_header=False,
    display_format='row'
)

# Display
display(predictor_lrs.display_results(results_lrs_all, style='formatted'))

```

### Spectral translation

Predict Gaia XP spectrum:
```python
import json
from spectral_retrieval import SpectralRetriever
from predict_lrs_wclip_v0 import load_spectrum_data

# Configuration
with open('config_retrieval.json', 'r') as f:
    config = json.load(f)
retriever = SpectralRetriever(**config)

# Load the external spectra data
wavelength, flux = load_spectrum_data('./test_data/lrs/sample1_matrix.fits')

# Predict corresponding Gaia XP spectrum
prediction_external = retriever.predict_cross_modal(
    query_spectrum=(wavelength, flux),
    query_type='lamost_spectra'
)

# Plot
retriever.plot_cross_modal_prediction(
    prediction_external,
    save_path='./plots/external_lamost_to_gaia_prediction.png'
)

```

### Spectral Similarity Search

Find the top-4 most similar stars from Gaia XP catalog:
```python
# Download test data only
!python download_and_setup.py --test-data-only

# Build embedding database from test data
retriever.build_embedding_database(batch_size=1000, save_path='./test_embeddings.npz')

# Load external LAMOST spectrum
wavelength, flux = load_spectrum_data('./test_data/lrs/sample1_matrix.fits')

# Find similar Gaia XP spectra
results_external_cross = retriever.find_similar_spectra(
    query_spectrum=(wavelength, flux),
    query_type='lamost_spectra',
    search_type='cross_modal',
    top_k=4
)

# Plot
retriever.plot_retrieval_results(
    results_external_cross,
    save_path='./plots/external_lamost_to_gaia_cross.png'
)

```

---

## Citation

If you use **SpecCLIP** in your research, please cite the paper:

```bibtex
@ARTICLE{2025arXiv250701939Z,
       author = {{Zhao}, Xiaosheng and {Huang}, Yang and {Xue}, Guirong and {Kong}, Xiao and
                 {Liu}, Jifeng and {Tang}, Xiaoyu and {Beers}, Timothy C. and
                 {Ting}, Yuan-Sen and {Luo}, A-Li},
        title = "{SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics, Solar and Stellar Astrophysics,
                 Artificial Intelligence, Machine Learning},
         year = 2025,
        month = jul,
          eid = {arXiv:2507.01939},
        pages = {arXiv:2507.01939},
          doi = {10.48550/arXiv.2507.01939},
archivePrefix = {arXiv},
       eprint = {2507.01939},
 primaryClass = {astro-ph.IM},
}
```
---

## Acknowledgments

We acknowledge the open-source [AstroCLIP](https://github.com/PolymathicAI/AstroCLIP) code base, upon which this project is largely built. 

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/Xiaosheng-Zhao/SpecCLIP/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Xiaosheng-Zhao/SpecCLIP/discussions)
- **Email:** [xzhao113@jh.edu]

# 🌌 SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-SpecCLIP-yellow)](https://huggingface.co/astroshawn/SpecCLIP/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.01939-b31b1b.svg)](https://arxiv.org/abs/2507.01939)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xiaosheng-Zhao/SpecCLIP/blob/main/downstream-with-demo/colab_tutorial.ipynb)

**A foundation-like model for cross-survey stellar spectroscopy combining contrastive learning, domain-specific information preservation, and cross-modal prediction**

> **SpecCLIP** aligns low-resolution LAMOST spectra with Gaia XP photometric spectra using CLIP-style contrastive learning while preserving domain-specific information.  
> It learns a *general-purpose spectral representation* that enables **parameter estimation**, **spectral translation**, and **similarity retrieval** across different spectroscopic surveys.

**Key Capabilities:**
- Predict stellar parameters (T_eff, log g, [Fe/H], RV, varied chemical abundances, extinction, etc.) from LAMOST LRS and Gaia XP spectrum
- Translate between different spectroscopic surveys (LAMOST ⟷ Gaia XP)
- Retrieve similar stars from both LAMOST LRS spectra and Gaia XP spectra

**Model Architecture and Down-Stream Tasks Overview (click for full image)**

<p align="center">
    <img width="1000" src="./img/SpecCLIP_model_with_analysis_card.png"/>
</p>

---

## Quick Start

### Try it in Colab — no install required

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xiaosheng-Zhao/SpecCLIP/blob/main/downstream-with-demo/colab_tutorial_full.ipynb)

Click the badge to open the tutorial directly in Google Colab — no local setup required. The notebook clones this repo, installs dependencies, downloads the pretrained checkpoints (~4.5 GB), and walks through similar-star retrieval and cross-modal spectrum prediction end-to-end.

What you need:
- A GPU runtime is **optional**. A Colab GPU (`Runtime → Change runtime type → T4 GPU`) speeds up the embedding-build step, but the demo runs fine on CPU too — either pick a CPU runtime, or pass `device='cpu'` when constructing the retriever (an optional commented line is included in the notebook).

> Prefer to run locally? Follow the **Installation** steps below and open [`downstream-with-demo/demo_download_v0.ipynb`](./downstream-with-demo/demo_download_v0.ipynb) — same content, no `git clone` / `pip install` cells.

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

| Model                | Description                                          | Embedding Dim | Param | Link                                                                       | Config | 
| -------------------- | ---------------------------------------------------- | ------------- | -------------------------------------------------------------------------- | ------------------------- |-------- |
| `SpecCLIP-LRS`      | LAMOST LRS masked transformer    | 768           |43M | [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/encoders/lrs_encoder.ckpt) | [config](./config/specformer_lrs_mt.yaml) | 
| `SpecCLIP-XP` | Gaia XP  masked transformer  | 768           |43M | [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/encoders/xp_encoder.ckpt) | [config](./config/specformer_xp_mt.yaml) | 
| `SpecCLIP-XP-oae` | Gaia XP  ordinary auto-encoder  | 768           |43M| [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/encoders/xp_encoder_mlp.ckpt) |[config](./config/specformer_xp_oae.yaml) | 
| `SpecCLIP-base` | Gaia XP  ⟷ LAMOST contrastive  | 768           |100M| [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/specclip/specclip_model_base.ckpt) |[config](./config/specclip_base_lrs_mt_xp_oae.yaml) |
| `SpecCLIP-pr` | Gaia XP  ⟷ LAMOST contrastive +pred+recon  | 768           |168M| [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/specclip/specclip_model_predrecon_mlp.ckpt) |[config](./config/specclip_pr_lrs_mt_xp_oae.yaml) |
| `SpecCLIP-split` | Gaia XP  ⟷ LAMOST contrastive+(pred+recon-split)    | 768           |126M| [🤗 Hugging Face](https://huggingface.co/astroshawn/SpecCLIP/blob/main/specclip/specclip_model_split_mlp.ckpt) |[config](./config/specclip_split_lrs_mt_xp_oae.yaml) |

**Examples of training with your own data:**  
Refer to the `scripts/` directory to train different model variants. Before running any training script, create a `.env` file inside the `specclip/` folder and include:

```bash
SPECCLIP_ROOT="/path/to/your/specclip"
WANDB_ENTITY_NAME="your_wandb_entity"
```
put the data (refer to [datamodule](./specclip/data/datamodule.py) for data structrue example) in `{SPECCLIP_ROOT}/data`, and update the data file name in `config` files.

---

## Usage Examples
See the full demo here: [Full Demo](./downstream-with-demo/demo_download_v0.ipynb)
### Download Model and Data at once
```bash
!pip install -q huggingface_hub
from huggingface_hub import login

# Run in terminal to login huggingface: huggingface-cli login

# Set the path of LOCAL_MODEL_DIR in download_and_setup.py and download model/test data
!python download_and_setup.py 
``` 

### Spectral Translation

Predict Gaia XP spectrum from LAMOST LRS:
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

### Parameter Prediction

**Coming soon.**  
This section will include examples of using SpecCLIP embeddings with downstream models (e.g., MLP, SBI) for stellar-parameter prediction.

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

We acknowledge the [AstroCLIP](https://github.com/PolymathicAI/AstroCLIP) open-source code base, on which this project is substantially built.

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/Xiaosheng-Zhao/SpecCLIP/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Xiaosheng-Zhao/SpecCLIP/discussions)
- **Email:** [xzhao113@jh.edu]

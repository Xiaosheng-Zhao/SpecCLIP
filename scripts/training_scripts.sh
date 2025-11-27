# Here are a collections of training scripts that produce the models in the paper.

# Scrits run under the parent folder, `wandb login` if using `WANDB_MODE=online`.

# 1. Gaia XP pretrained model, using the ordinary auto-encoder (OAE) model:
WANDB_MODE=online torchrun --nproc_per_node=8 specclip/trainer.py fit -c ./config/specformer_xp_oae.yaml

# 2. Gaia XP pretrained model, using the masked transformer (MT) model:
WANDB_MODE=online torchrun --nproc_per_node=8 specclip/trainer.py fit -c ./config/specformer_xp_mt.yaml

# 3. LAMOST LRS pretrained model, using the masked transformer (MT) model:
WANDB_MODE=online torchrun --nproc_per_node=8 specclip/trainer.py fit -c ./config/specformer_lrs_mt.yaml

# 4. based specclip model, using only contrastive loss:
WANDB_MODE=online torchrun --nproc_per_node=8 specclip/trainer.py fit -c ./config/specclip_base_lrs_mt_xp_oae.yaml

# 5. speclilp-pr model: contrastive loss + reconstruction loss + prediction loss, using a unified embedding vector for each modality
WANDB_MODE=online torchrun --nproc_per_node=8 specclip/trainer.py fit -c ./config/specclip_pr_lrs_mt_xp_oae.yaml

# 6a. specclip-split model: contrastive loss + reconstruction loss + prediction loss, using split (shared+non-shared) embedding vectors for each modality.
WANDB_MODE=online torchrun --nproc_per_node=8 specclip/trainer.py fit -c ./config/specclip_split_lrs_mt_xp_oae.yaml

# 6b. specclip-split model: contrastive loss + reconstruction loss + prediction loss, using split (shared+non-shared) embedding vectors for each modality. 
WANDB_MODE=online torchrun --nproc_per_node=8 specclip/trainer.py fit -c ./config/specclip_split_lrs_mt_xp_oae.yaml
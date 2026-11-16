# Mesh (MeshGPT)

This folder contains the mesh code. It builds on the [MeshGPT](https://github.com/audi/MeshGPT) code base and adds the hyper-autoencoder between MeshGPT's graph encoder and its vector quantizer. The MeshGPT encoder, quantizer and decoder are loaded from the MeshGPT authors' pretrained checkpoint and kept frozen. Only the hyper-autoencoder is trained.

All commands below are run from this directory.

## Setup

The code was run with Python 3.10, PyTorch 2.5.1 and CUDA 12.1.

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"
pip install -r requirements.txt
```

## Data and pretrained checkpoint

1. Download the MeshGPT authors' data and pretrained models from the link in the [MeshGPT README](https://github.com/audi/MeshGPT#pre-trained-models-and-data).
2. Place the ShapeNet data at `data/shapenet/processed_data.pkl` and the pretrained vocabulary checkpoint at `pretrained/vocabulary/checkpoints/749-0.ckpt`.
3. Build the train/val split used in the paper. The released split has [val shapes that also appear in train](https://github.com/audi/MeshGPT/issues/10), so this script re-splits `processed_data.pkl` by shape ID using `data/splits/train_subset.lst` and `data/splits/val_subset.lst`:

```bash
python dataset/repickle_clean_split.py \
    --pickle data/shapenet/processed_data.pkl \
    --out data/shapenet/processed_data_clean.pkl
```

## Training

Fixed-length pretraining: the hyper-autoencoder is trained from the MeshGPT checkpoint.

```bash
python trainer/train_vocabulary.py --config-name shapenet-v2 max_epoch=2000 val_check_interval=10 only_chairs=true batch_size=8 gradient_accumulation_steps=8 num_val_samples=4 use_sequence_distillation=true num_sequence_tokens=256 use_nested_dropout=false n_encoder_self_attn_blocks=8 encoder_zero_init=true encoder_use_rope=true query_binding=rope use_causal_attention=true distill_before_vq=true n_decoder_self_attn_blocks=8 use_order_preserving_ae=true encoder_use_pos_embed=true freeze_base=true ae_only_loss=false code_cls_weight=1.0 ae_recon_weight=1.0 ae_warmup_steps=0 dataset_root=data/shapenet/processed_data_clean.pkl vq_resume=pretrained/vocabulary/checkpoints/749-0.ckpt
```

Then, we follow by training with suffix dropout with a sampled number of kept tokens, from the end of stage 1.

```bash
python trainer/train_vocabulary.py --config-name shapenet-v2 max_epoch=2000 val_check_interval=10 only_chairs=true batch_size=8 gradient_accumulation_steps=8 num_val_samples=4 use_sequence_distillation=true num_sequence_tokens=256 use_nested_dropout=true nested_dropout_size_sampling=uniform n_encoder_self_attn_blocks=8 encoder_zero_init=false encoder_use_rope=true query_binding=rope use_causal_attention=true distill_before_vq=true n_decoder_self_attn_blocks=8 use_order_preserving_ae=true encoder_use_pos_embed=true freeze_base=true ae_only_loss=false code_cls_weight=1.0 ae_recon_weight=1.0 ae_warmup_steps=0 dataset_root=data/shapenet/processed_data_clean.pkl vq_resume=runs/<stage 1 run>/checkpoints/1999-0.ckpt
```

## License

This folder is derived from MeshGPT and is distributed under the MeshGPT license in [`LICENSE`](LICENSE).

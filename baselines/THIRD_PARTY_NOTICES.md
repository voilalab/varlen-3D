# Third-party code and licenses

This repository does not contain any upstream repository. `setup_methods.sh` clones each one from
its public source at a pinned commit, so each checkout comes with its own license. What this
repository does contain is:

- `patches/<method>.patch`: our modifications to upstream files, as unified diffs. These include
  some lines of the upstream code as context.
- `overlays/<method>/`: new files we wrote. They are added to the upstream checkout and call into
  it. The one exception is `overlays/MeshGPT_old/config.yaml`, which is derived from MeshGPT's
  config (see below).

Each patch file starts with a header naming its upstream project, copyright holder and license.

| Method | Upstream (pinned commit) | Upstream license | Covers in this repo | License copy |
|---|---|---|---|---|
| CONs | [autonomousvision/convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks) @ `838bea5` | MIT, © 2020 Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, Andreas Geiger | `patches/convolutional_occupancy_networks.patch` | [`third_party_licenses/convolutional_occupancy_networks.LICENSE`](third_party_licenses/convolutional_occupancy_networks.LICENSE) |
| Instant-NGP | [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) @ `018dea6` | NVIDIA Source Code License for instant neural graphics primitives (**non-commercial**), © 2022-2025 NVIDIA Corporation & affiliates | `patches/instant-ngp.patch` | [`third_party_licenses/instant-ngp.LICENSE.txt`](third_party_licenses/instant-ngp.LICENSE.txt) |
| MeshGPT (original) | [audi/MeshGPT](https://github.com/audi/MeshGPT) @ `028648f` | Automotive Development Public Non-Commercial License v1.0 (ADPNCL, **non-commercial**), © 2024 Audi AG | `patches/MeshGPT_old.patch`, `overlays/MeshGPT_old/config.yaml` | [`third_party_licenses/MeshGPT.LICENSE`](third_party_licenses/MeshGPT.LICENSE) |
| NGF | [iveevi/ngf](https://github.com/iveevi/ngf) @ `1cfe0b8` | none published | `patches/ngf.patch` | n/a |
| BACON | [computational-imaging/bacon](https://github.com/computational-imaging/bacon) @ `c691c69` | none published | `patches/bacon.patch` | n/a |

## Notes per license

**Instant-NGP and MeshGPT are non-commercial only.** The patches for them, and any checkout built
with those patches applied, may be used for research and other non-commercial purposes only.

**MeshGPT (ADPNCL).** The modifications in `patches/MeshGPT_old.patch` and
`overlays/MeshGPT_old/config.yaml` are derived from the MeshGPT code provided by Audi AG and are
distributed under the ADPNCL v1.0. The patch file documents the differences from the original
and the dates the modifications were created (ADPNCL section 3.4(a)). Section 3.4(b) also asks
whoever modifies the code to send that documentation to the Contributor (Audi AG,
ai-licenses@audi.de).

The MeshGPT pretrained checkpoint (`749-0.ckpt`) is not included here. It is downloaded from the
authors' release under their terms; see [`docs/meshgpt_original.md`](docs/meshgpt_original.md).

**NGF and BACON.** Neither upstream repository publishes a license, so all rights to their code
remain with their authors. This repository does not redistribute either codebase: the patches only
make small, targeted changes to a checkout that you obtain from the authors' public repository.
These two patches are provided only so that the benchmark can be reproduced for research
purposes.

## Datasets and pretrained models

- **ShapeNet** (ShapeNetCore v2) is not included. It is obtained from
  [shapenet.org](https://shapenet.org) under its terms of use.
- **Occupancy Networks preprocessed data** (used by CONs) is downloaded by the patched
  `scripts/download_data.sh` from the Occupancy Networks authors.

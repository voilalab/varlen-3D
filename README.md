# Data-Driven Variable-Length Compression of 3D Representations

Code for the paper. Each folder has its own README with setup and commands.

| Folder | Contents |
|---|---|
| [`meshgpt/`](meshgpt/README.md) | Our method on meshes: the hyper-autoencoder trained on a frozen, pretrained MeshGPT. |
| [`SDF/`](SDF/README.md) | Our method on signed distance fields: the hyper-autoencoder trained on a frozen, pretrained VecSet autoencoder. |
| [`baselines/`](baselines/README.md) | The baselines (Instant-NGP, NGF, BACON, Convolutional Occupancy Networks, original MeshGPT), the shared Chamfer distance and compression-ratio code, and the rate-distortion figure scripts. |

## License

The repository is under the BSD-2 license in [`LICENSE`](LICENSE), except:
- `meshgpt/` is derived from MeshGPT and is under the MeshGPT license in [`meshgpt/LICENSE`](meshgpt/LICENSE).
- `baselines/` has its own [`LICENSE`](baselines/LICENSE) and [`THIRD_PARTY_NOTICES.md`](baselines/THIRD_PARTY_NOTICES.md).

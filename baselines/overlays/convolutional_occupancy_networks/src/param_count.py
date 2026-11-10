#

def _as_plane_type_list(plane_type: Any) -> List[Any]:
    if plane_type is None:
        return []
    if isinstance(plane_type, (list, tuple)):
        return list(plane_type)
    return [plane_type]


def non_shared_latent_param_count(cfg: dict) -> int:
    """Return the number of scalar features in the latent grid/planes (per mesh).

    Raises:
        ValueError: if plane_type / resolutions cannot be interpreted.
    """
    model = cfg.get("model") or {}
    enc_kw = model.get("encoder_kwargs") or {}
    c_dim = int(model.get("c_dim", 0))
    if c_dim <= 0:
        raise ValueError("model.c_dim must be a positive int in config")

    plane_type = enc_kw.get("plane_type", "xz")
    pt = _as_plane_type_list(plane_type)
    if not pt:
        raise ValueError("encoder_kwargs.plane_type is missing or empty")

    plane_res = enc_kw.get("plane_resolution")
    grid_res = enc_kw.get("grid_resolution")

    total = 0

    if any(p == "grid" for p in pt):
        if grid_res is None:
            raise ValueError(
                "plane_type includes 'grid' but encoder_kwargs.grid_resolution is missing"
            )
        g = int(grid_res)
        total += (g**3) * c_dim

    for plane in ("xz", "xy", "yz"):
        if any(p == plane for p in pt):
            if plane_res is None:
                raise ValueError(
                    f"plane_type includes '{plane}' but encoder_kwargs.plane_resolution is missing"
                )
            r = int(plane_res)
            total += (r**2) * c_dim

    if total <= 0:
        raise ValueError(
            f"Could not derive non-shared count from plane_type={pt!r}; "
            "expected 'grid' and/or one of xz, xy, yz."
        )

    return int(total)


def total_checkpoint_params(state_dict: dict) -> int:
    """All tensor elements in a checkpoint state_dict (shared + non-shared weights)."""
    import torch

    return sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))

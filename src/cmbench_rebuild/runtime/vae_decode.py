"""Decode every causal step, retaining only requested output pixels on CPU.

The pinned Wan VAE repeatedly concatenates its entire decoded history on GPU.
We use the same scaling, convolution and stateful decoder calls, but discard
context RGB once its decoder state has been updated. Vendor code stays intact.
"""
from __future__ import annotations


def decode_continuation(vae, latent, pixel_slice):
    import torch

    start, stop = pixel_slice.start, pixel_slice.stop
    if start is None or stop is None or not 0 <= start < stop or pixel_slice.step not in (None, 1):
        raise ValueError("pixel_slice must be a nonempty forward interval")
    model = vae.model
    model.clear_cache()
    retained, decoded_frames = [], 0
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=vae.dtype,
                enabled=latent.is_cuda and vae.dtype != torch.float32):
            z = latent.unsqueeze(0)
            mean, inverse_std = vae.scale
            if isinstance(mean, torch.Tensor):
                z = z / inverse_std.view(1, model.z_dim, 1, 1, 1) + mean.view(1, model.z_dim, 1, 1, 1)
            else:
                z = z / inverse_std + mean
            x = model.conv2(z)
            for index in range(x.shape[2]):
                model._conv_idx = [0]
                frames = model.decoder(x[:, :, index:index + 1],
                    feat_cache=model._feat_map, feat_idx=model._conv_idx)
                next_frame = decoded_frames + frames.shape[2]
                left, right = max(start, decoded_frames), min(stop, next_frame)
                if left < right:
                    selected = frames[0, :, left - decoded_frames:right - decoded_frames]
                    retained.append(selected.float().clamp(-1, 1).cpu())
                decoded_frames = next_frame
        if decoded_frames < stop:
            raise RuntimeError(f"VAE timeline is too short: decoded={decoded_frames}, required={stop}")
        return torch.cat(retained, dim=1), decoded_frames
    finally:
        model.clear_cache()

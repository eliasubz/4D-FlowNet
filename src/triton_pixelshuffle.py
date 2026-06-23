from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _pixelshuffle3d_kernel(
        x_ptr,
        y_ptr,
        stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_yb, stride_yc, stride_yd, stride_yh, stride_yw,
        B, OC, D, H, W,
        r,
        NUM_ELEMENTS,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < NUM_ELEMENTS
        
        # Output spatial dims
        OD = D * r
        OH = H * r
        OW = W * r
        
        # Decompose 1D output offset into 5D coordinates (b, oc, od, oh, ow)
        ow = offsets % OW
        temp = offsets // OW
        oh = temp % OH
        temp = temp // OH
        od = temp % OD
        temp = temp // OD
        oc = temp % OC
        b = temp // OC
        
        # Compute input coordinates (id, ih, iw) and sub-pixel offsets (rd, rh, rw)
        id = od // r
        rd = od % r
        
        ih = oh // r
        rh = oh % r
        
        iw = ow // r
        rw = ow % r
        
        # Compute input channel index ic corresponding to (oc, rd, rh, rw)
        # Match PyTorch permute(0, 1, 5, 2, 6, 3, 7, 4) mapping
        ic = oc * (r * r * r) + rd * (r * r) + rh * r + rw
        
        # Compute 1D pointer offsets
        x_offset = b * stride_xb + ic * stride_xc + id * stride_xd + ih * stride_xh + iw * stride_xw
        y_offset = b * stride_yb + oc * stride_yc + od * stride_yd + oh * stride_yh + ow * stride_yw
        
        # Memory transfer
        x_vals = tl.load(x_ptr + x_offset, mask=mask)
        tl.store(y_ptr + y_offset, x_vals, mask=mask)


    def triton_pixel_shuffle_3d(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
        """Rearrange channels into spatial dimensions using a custom Triton kernel.

        Args:
            x: Input tensor shaped [B, C, D, H, W], where C must be divisible by upscale_factor^3.
            upscale_factor: Upscaling ratio (r).

        Returns:
            Output tensor shaped [B, C // r^3, D * r, H * r, W * r].
        """
        assert x.is_cuda, "Input tensor must be on CUDA to use Triton!"
        b, c, d, h, w = x.shape
        r = upscale_factor
        
        assert c % (r ** 3) == 0, f"Channel count {c} is not divisible by upscale_factor^3 ({r**3})!"
        oc = c // (r ** 3)
        
        od, oh, ow = d * r, h * r, w * r
        y = torch.empty((b, oc, od, oh, ow), device=x.device, dtype=x.dtype)
        
        num_elements = y.numel()
        
        # Define grid size
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        
        _pixelshuffle3d_kernel[grid](
            x, y,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
            b, oc, d, h, w,
            r,
            num_elements,
            BLOCK_SIZE=1024
        )
        return y
else:
    def triton_pixel_shuffle_3d(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
        raise ImportError("Triton is not available or could not be loaded in this environment.")

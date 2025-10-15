import numpy as np
import torch
import torch.fft

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available.")


@triton.jit
def frc_forward_kernel(
    # inputs
    fft1_r, fft1_i, fft2_r, fft2_i,
    radial_f, freq_bins,
    # outputs
    out_frc, num_r_out, num_i_out, d1_out, d2_out,
    B, H, W, NR,
    st_b, st_h, st_w,
    st_rh, st_rw,
    st_ob, st_or,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)          # batch dim
    r = tl.program_id(1)          # ring  dim
    if b >= B or r >= NR:
        return

    # ----------- load ring boundaries & init acc -----------------------------
    f_lo = tl.load(freq_bins + r)
    f_hi = tl.load(freq_bins + r + 1)

    num_r = tl.zeros((), tl.float32)
    num_i = tl.zeros((), tl.float32)
    d1 = tl.zeros((), tl.float32)
    d2 = tl.zeros((), tl.float32)

    # ptr bases
    base1_r = fft1_r + b * st_b
    base1_i = fft1_i + b * st_b
    base2_r = fft2_r + b * st_b
    base2_i = fft2_i + b * st_b

    for h in range(0, H):
        # pointer offset for this row
        row_off = h * st_h
        # broadcast radial-f of this row
        rh_ptr = radial_f + h * st_rh

        for w0 in range(0, W, BLOCK):
            offs = w0 + tl.arange(0, BLOCK)
            mask_w = offs < W

            # radial frequency of BLOCK pixels
            rf = tl.load(rh_ptr + offs * st_rw,
                         mask=mask_w, other=0.0)

            m_ring = (rf >= f_lo) & (rf < f_hi) & mask_w
            if tl.sum(m_ring) > 0:
                # pixel pointers
                pix_off = row_off + offs * st_w

                x_r = tl.load(base1_r + pix_off, mask=m_ring, other=0.0)
                x_i = tl.load(base1_i + pix_off, mask=m_ring, other=0.0)
                y_r = tl.load(base2_r + pix_off, mask=m_ring, other=0.0)
                y_i = tl.load(base2_i + pix_off, mask=m_ring, other=0.0)

                # cross corr  x * conj(y)
                cr = x_r * y_r + x_i * y_i
                ci = x_i * y_r - x_r * y_i
                num_r += tl.sum(cr)
                num_i += tl.sum(ci)

                # magnitudes
                mag1 = x_r * x_r + x_i * x_i
                mag2 = y_r * y_r + y_i * y_i
                d1 += tl.sum(mag1)
                d2 += tl.sum(mag2)

    # ------------- final frc for (b,r) ---------------------------------------
    frc_val = 0.0
    if d1 > 1e-9 and d2 > 1e-9:
        n_abs = tl.sqrt(num_r * num_r + num_i * num_i)
        denom = tl.sqrt(d1 * d2)
        frc_val = n_abs / denom if denom > 1e-9 else 0.0

    # write outputs
    o_ptr = out_frc + b*st_ob + r*st_or
    tl.store(o_ptr, frc_val)

    tl.store(num_r_out + b*st_ob + r*st_or, num_r)
    tl.store(num_i_out + b*st_ob + r*st_or, num_i)
    tl.store(d1_out    + b*st_ob + r*st_or, d1)
    tl.store(d2_out    + b*st_ob + r*st_or, d2)


@triton.jit
def frc_backward_kernel(
    # forward inputs
    fft1_r, fft1_i, fft2_r, fft2_i,
    radial_f, freq_bins,
    num_r_in, num_i_in, d1_in, d2_in,
    # grad from upstream
    g_out,
    # out-grads (freq domain, real / imag)
    g1_r, g1_i, g2_r, g2_i,
    B, H, W, NR,
    st_b, st_h, st_w,
    st_rh, st_rw,
    st_ob, st_or,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    r = tl.program_id(1)
    if b >= B or r >= NR:
        return

    # ----- scalars for this (b,r) -------------------------------------------
    Nr = tl.load(num_r_in + b*st_ob + r*st_or)
    Ni = tl.load(num_i_in + b*st_ob + r*st_or)
    D1 = tl.load(d1_in    + b*st_ob + r*st_or)
    D2 = tl.load(d2_in    + b*st_ob + r*st_or)
    go = tl.load(g_out    + b*st_ob + r*st_or)

    n_abs = tl.sqrt(Nr*Nr + Ni*Ni)
    denom = tl.sqrt(D1*D2)
    valid = (n_abs > 1e-9) & (D1 > 1e-9) & (D2 > 1e-9) & (denom > 1e-9)
    if ~valid:
        return

    c1_r = go * Nr / (n_abs * denom)
    c1_i = go * Ni / (n_abs * denom)
    c2   = go * n_abs / (denom * D1)
    c3   = go * n_abs / (denom * D2)

    f_lo = tl.load(freq_bins + r)
    f_hi = tl.load(freq_bins + r + 1)

    # ptr bases
    base1_r = fft1_r + b * st_b
    base1_i = fft1_i + b * st_b
    base2_r = fft2_r + b * st_b
    base2_i = fft2_i + b * st_b

    out1_r = g1_r + b * st_b
    out1_i = g1_i + b * st_b
    out2_r = g2_r + b * st_b
    out2_i = g2_i + b * st_b

    for h in range(0, H):
        row_off = h * st_h
        rh_ptr = radial_f + h * st_rh

        for w0 in range(0, W, BLOCK):
            offs = w0 + tl.arange(0, BLOCK)
            mask_w = offs < W

            rf = tl.load(rh_ptr + offs * st_rw,
                         mask=mask_w, other=0.0)
            m_ring = (rf >= f_lo) & (rf < f_hi) & mask_w
            if tl.sum(m_ring) > 0:
                pix_off = row_off + offs * st_w
                a = tl.load(base1_r + pix_off, mask=m_ring, other=0.0)
                b_ = tl.load(base1_i + pix_off, mask=m_ring, other=0.0)
                c = tl.load(base2_r + pix_off, mask=m_ring, other=0.0)
                d = tl.load(base2_i + pix_off, mask=m_ring, other=0.0)

                # -------- grad wrt x = a+ib --------------------------------------
                t1r =  c * c1_r + d * c1_i      # Re{ conj(y_k) * c1 }
                t1i = -d * c1_r + c * c1_i      # Im{ conj(y_k) * c1 } ; Note: this is c*c1_i - d*c1_r
                gx_r = t1r - a * c2
                # gx_i = t1i - b_ * c2          # ORIGINAL LINE
                gx_i = -t1i - b_ * c2         # MODIFIED LINE

                # -------- grad wrt y = c+id --------------------------------------
                t2r =  a * c1_r - b_ * c1_i     # Re{ x_k * c1 }
                t2i =  a * c1_i + b_ * c1_r     # Im{ x_k * c1 }
                gy_r = t2r - c * c3
                # gy_i = -t2i + d * c3          # USER'S CURRENT LINE (from previous fix)
                                                # Based on the same logic as gx_i, this should be:
                gy_i = -t2i - d * c3          # CONSISTENT FIX if y also needs it

                tl.store(out1_r + pix_off, gx_r, mask=m_ring)
                tl.store(out1_i + pix_off, gx_i, mask=m_ring)
                tl.store(out2_r + pix_off, gy_r, mask=m_ring)
                tl.store(out2_i + pix_off, gy_i, mask=m_ring)

################################################################################
#                autograd-friendly Python wrapper                              #
################################################################################
class TritonFRC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1: torch.Tensor, img2: torch.Tensor,
                num_rings: int = 20, block: int = 128):
        if img1.shape != img2.shape or img1.ndim != 3:
            raise ValueError("Expect (B,H,W) identical shapes")

        B, H, W = img1.shape
        device = img1.device
        dtype = img1.dtype

        # FFT (complex64/128) → shift → split
        f1 = torch.fft.fftshift(torch.fft.fft2(img1), dim=(-2, -1))
        f2 = torch.fft.fftshift(torch.fft.fft2(img2), dim=(-2, -1))
        f1_r, f1_i = f1.real.contiguous(), f1.imag.contiguous()
        f2_r, f2_i = f2.real.contiguous(), f2.imag.contiguous()

        # radial frequency grid
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, device=device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, device=device))
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radial = torch.sqrt(xx**2 + yy**2).contiguous()

        # bins & ring-mid frequencies
        pos = radial[radial > 1e-9]
        f_min = pos.min() if pos.numel() else 1e-6
        f_max = radial.max()
        bins = torch.linspace(f_min, f_max, num_rings + 1,
                              device=device, dtype=radial.dtype)
        ring_mid = 0.5 * (bins[:-1] + bins[1:])  # (NR,)

        # output+cache tensors
        frc = torch.zeros(B, num_rings, device=device, dtype=dtype)
        nr_r = torch.empty_like(frc)
        nr_i = torch.empty_like(frc)
        d1   = torch.empty_like(frc)
        d2   = torch.empty_like(frc)

        grid = (B, num_rings)
        frc_forward_kernel[grid](
            f1_r, f1_i, f2_r, f2_i,
            radial, bins,
            frc, nr_r, nr_i, d1, d2,
            B, H, W, num_rings,
            f1_r.stride(0), f1_r.stride(1), f1_r.stride(2),
            radial.stride(0), radial.stride(1),
            frc.stride(0),   frc.stride(1),
            BLOCK=block,
        )

        # save for backward
        ctx.save_for_backward(f1_r, f1_i, f2_r, f2_i,
                              radial, bins, nr_r, nr_i, d1, d2)
        ctx.dims = (B, H, W, num_rings, block)
        return frc, ring_mid

    @staticmethod
    def backward(ctx, g_frc, g_mid=None):
        (f1_r, f1_i, f2_r, f2_i,
         radial, bins, nr_r, nr_i, d1, d2) = ctx.saved_tensors
        B, H, W, NR, block = ctx.dims
        dtype = f1_r.dtype
        device = f1_r.device

        # allocate freq-domain gradients
        g1_r = torch.zeros_like(f1_r)
        g1_i = torch.zeros_like(f1_i)
        g2_r = torch.zeros_like(f2_r)
        g2_i = torch.zeros_like(f2_i)

        frc_backward_kernel[(B, NR)](
            f1_r, f1_i, f2_r, f2_i,
            radial, bins,
            nr_r, nr_i, d1, d2,
            g_frc.contiguous(),
            g1_r, g1_i, g2_r, g2_i,
            B, H, W, NR,
            f1_r.stride(0), f1_r.stride(1), f1_r.stride(2),
            radial.stride(0), radial.stride(1),
            g_frc.stride(0), g_frc.stride(1),
            BLOCK=block,
        )

        # freq->space  (dF/dx_freq  → dF/dx_img)
        def freq2img(gr, gi):
            g_complex = torch.complex(gr, gi)
            g_shift   = torch.fft.ifftshift(g_complex, dim=(-2, -1))
            g_img     = torch.fft.ifft2(g_shift).real
            return g_img.to(dtype)

        gx = freq2img(g1_r, g1_i)
        gy = freq2img(g2_r, g2_i)

        return gx, gy, None, None   # None for num_rings & block


################################################################################
#                              convenience API                                #
################################################################################
def calculate_frc_triton(img1: torch.Tensor, img2: torch.Tensor, num_rings: int = 20, block: int = 128):
    """Returns (frc_curve, ring_mid_freqs)"""
    return TritonFRC.apply(img1, img2, num_rings, block)


def calculate_frc_pytorch(batch_image1: torch.Tensor, batch_image2: torch.Tensor, num_rings: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of FRC calculation (optimized version).
    """
    if batch_image1.shape != batch_image2.shape:
        raise ValueError("Input image batches must have the same dimensions.")
    if batch_image1.ndim != 3 or batch_image2.ndim != 3:
        raise ValueError("Input images must be 3D tensors (B, H, W).")

    B, H, W = batch_image1.shape
    device = batch_image1.device
    dtype = batch_image1.dtype

    # Create frequency grid (common for all images in the batch)
    y_freq = torch.fft.fftshift(torch.fft.fftfreq(H, device=device))
    x_freq = torch.fft.fftshift(torch.fft.fftfreq(W, device=device))
    yy, xx = torch.meshgrid(y_freq, x_freq, indexing='ij')
    radial_freq = torch.sqrt(xx**2 + yy**2)

    # Determine frequency bins (rings)
    max_freq = radial_freq.max()
    positive_freqs = radial_freq[radial_freq > 1e-9]
    actual_min_freq = positive_freqs.min() if positive_freqs.numel() > 0 else 1e-6

    freq_bins = torch.linspace(actual_min_freq, max_freq, num_rings + 1, device=device)
    
    common_ring_frequencies = torch.zeros(num_rings, device=device, dtype=dtype)
    for i in range(num_rings):
        common_ring_frequencies[i] = (freq_bins[i] + freq_bins[i + 1]) / 2

    # Compute 2D Fourier Transforms for the entire batch
    fft_batch1 = torch.fft.fftshift(torch.fft.fft2(batch_image1), dim=(-2, -1))
    fft_batch2 = torch.fft.fftshift(torch.fft.fft2(batch_image2), dim=(-2, -1))

    batch_frc_curves = torch.zeros(B, num_rings, device=device, dtype=dtype)

    for i in range(num_rings):
        lower_bound = freq_bins[i]
        upper_bound = freq_bins[i+1]
        
        mask = (radial_freq >= lower_bound) & (radial_freq < upper_bound)
        
        if not mask.any():
            batch_frc_curves[:, i] = torch.nan
            continue

        fft1_ring_batch = fft_batch1[:, mask]
        fft2_ring_batch = fft_batch2[:, mask]

        numerator_batch = torch.sum(fft1_ring_batch * torch.conj(fft2_ring_batch), dim=1)
        denominator1_sq_batch = torch.sum(torch.abs(fft1_ring_batch)**2, dim=1)
        denominator2_sq_batch = torch.sum(torch.abs(fft2_ring_batch)**2, dim=1)
        
        current_frc_values = torch.full((B,), torch.nan, device=device, dtype=dtype)
        valid_denominators_mask = (denominator1_sq_batch > 1e-9) & (denominator2_sq_batch > 1e-9)
        
        if valid_denominators_mask.any():
            sqrt_denominators = torch.sqrt(denominator1_sq_batch[valid_denominators_mask] * denominator2_sq_batch[valid_denominators_mask])
            current_frc_values[valid_denominators_mask] = torch.abs(numerator_batch[valid_denominators_mask]) / sqrt_denominators
        
        batch_frc_curves[:, i] = current_frc_values

    return batch_frc_curves, common_ring_frequencies

import torch
import pytest
from prismquant import fit_rotation, dynamic_asym_int4, RotationFactor, compact_wy, WYFactor
from prismquant.rotations import full_hadamard_rows


def test_anchor_mapping_roundtrip_wy_and_serialization(tmp_path):
    torch.manual_seed(9)
    x = torch.randn(256, 256)
    x[:, :3] *= 12
    factor = fit_rotation(x, rank=2)
    signs = torch.randint(0, 2, (256,)).float().mul(2).sub(1)
    y = factor.apply(x, signs)
    relative = (factor.transpose(y, signs) - x).norm() / x.norm()
    assert relative < 1e-6
    w, v = compact_wy(factor.reflectors, factor.active)
    torch.testing.assert_close(WYFactor(factor, w, v).apply(x, signs), y, rtol=1e-5, atol=8e-6)
    factor.save(tmp_path / 'factor.pt')
    loaded = RotationFactor.load(tmp_path / 'factor.pt', torch.device('cpu'))
    assert torch.equal(loaded.apply(x, signs), y)
    assert factor.anchor_error < 1e-6


@pytest.mark.parametrize('n', [128, 1536, 3584, 2560, 9728])
def test_hadamard_preserves_energy(n):
    torch.manual_seed(n)
    x = torch.randn(2, n)
    y = full_hadamard_rows(x, torch.ones(n))
    torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1), rtol=1e-6, atol=1e-5)


def test_fp16_quantizer_degenerate_underflow_and_metadata():
    x = torch.stack([torch.ones(128) * 3.25, torch.arange(128) * 1e-10,
                     torch.linspace(-3, 4, 128)])
    y, s, z, q = dynamic_asym_int4(x, 128)
    assert torch.isfinite(y).all()
    assert torch.equal(y[0], x[0])
    assert s.dtype == z.dtype == torch.float16 and q.dtype == torch.uint8
    assert s[1].item() == 1 and q.max() <= 15
    assert torch.equal(y, (q.float() * s.float().unsqueeze(-1) + z.float().unsqueeze(-1)).reshape_as(x))


def test_invalid_calibration_fails_before_fitting():
    with pytest.raises(ValueError): fit_rotation(torch.randn(8, 128), rank=2)
    with pytest.raises(ValueError): fit_rotation(torch.randn(8, 130))
    with pytest.raises(ValueError): fit_rotation(torch.full((8, 128), float('nan')))


def test_signed_permutation_fold_preserves_swiglu_rotation():
    import copy
    from prismquant.fused import FoldedR4, fold_swiglu_weights
    torch.manual_seed(13)
    f = fit_rotation(torch.randn(320,256),rank=2)
    signs = torch.randint(0,2,(256,)).float().mul(2).sub(1)
    folded = FoldedR4.from_factor(f,signs)
    gate,up = torch.nn.Linear(32,256,bias=False),torch.nn.Linear(32,256,bias=False)
    x=torch.randn(5,32)
    expected=f.apply(torch.nn.functional.silu(gate(x))*up(x),signs)
    fold_swiglu_weights(gate,up,folded)
    actual=folded.apply(torch.nn.functional.silu(gate(x))*up(x))
    assert (actual-expected).norm()/expected.norm()<1e-6

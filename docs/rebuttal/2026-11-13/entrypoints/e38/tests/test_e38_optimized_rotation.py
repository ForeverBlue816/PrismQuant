"""Check saved-factor algebra independently of the optional CUDA extension."""
import math
from pathlib import Path
import unittest

import torch

from e38.optimized_rotation import FastOriginalE14R4, FastOriginalE14R2
from nar.activation_experiments import RotationFactor
from nar.e12_wy import WYFactor, compact_wy


def dense_hadamard(value, *, scale):
    matrix = torch.ones(1,1,device=value.device,dtype=value.dtype)
    while matrix.shape[0] < value.shape[-1]:
        matrix = torch.cat((torch.cat((matrix,matrix),1),torch.cat((matrix,-matrix),1)),0)
    return (value @ matrix)*scale


class OptimizedSavedRotation(unittest.TestCase):
    def factor(self, n=256, k=3):
        generator=torch.Generator().manual_seed(9)
        reflectors=torch.randn(k,n,generator=generator)
        reflectors/=reflectors.norm(dim=1,keepdim=True)
        return RotationFactor(n=n,b=128,reflectors=reflectors,active=torch.tensor([True,False,True]),
            source_order=torch.randperm(n,generator=generator),target_order=torch.randperm(n,generator=generator),anchor_error=0.)

    def test_r4_wy_gather_sign_and_hadamard_match_dense_composition(self):
        factor=self.factor();w,y=compact_wy(factor.reflectors,factor.active)
        signs=torch.where(torch.arange(factor.n)%3==0,-1.,1.)
        value=torch.linspace(-2,2,3*2*factor.n).reshape(3,2,factor.n).transpose(0,1)
        fast=FastOriginalE14R4(WYFactor(factor,w,y),signs,hadamard_fn=dense_hadamard)
        # Explicit ordered Householder matrices independently verify orientation.
        matrix=torch.eye(factor.n)
        for index in (0,2):
            v=factor.reflectors[index]
            matrix=matrix@(torch.eye(factor.n)-2*v[:,None]*v[None,:])
        rows=value.reshape(-1,factor.n)@matrix
        permuted=torch.empty_like(rows)
        permuted[:,factor.target_order]=rows[:,factor.source_order]
        expected=dense_hadamard((permuted*signs).reshape(-1,128),scale=1/math.sqrt(128)).reshape_as(value)
        self.assertTrue(torch.allclose(fast(value),expected,atol=2e-5,rtol=2e-5))

    def test_r2_keeps_ordered_active_reflectors_and_original_output_dtype(self):
        factor=self.factor();signs=torch.ones(factor.n);signs[::2]=-1
        value=torch.randn(1,7,factor.n,generator=torch.Generator().manual_seed(3)).half()
        fast=FastOriginalE14R2(factor,signs,hadamard_fn=dense_hadamard)
        result=fast(value);expected=factor.apply(value,signs)
        self.assertEqual(result.dtype,torch.float32)
        self.assertEqual(fast.active_indices,(0,2))
        self.assertTrue(torch.allclose(result,expected,atol=2e-6,rtol=2e-6))

    def test_probe_is_once_only_and_does_not_change_outputs(self):
        factor=self.factor();w,y=compact_wy(factor.reflectors,factor.active);audit=[]
        fast=FastOriginalE14R4(WYFactor(factor,w,y),torch.ones(factor.n),
            hadamard_fn=dense_hadamard,audit=audit,site='test.r4')
        value=torch.randn(12,factor.n,generator=torch.Generator().manual_seed(4))
        first=fast(value);second=fast(value)
        self.assertTrue(torch.equal(first,second));self.assertEqual(len(audit),1)
        self.assertEqual(audit[0]['rows'],8)
        self.assertFalse(audit[0]['numerical_execution_gate'])

    def test_actual_saved_e14_k8_factor(self):
        path=Path('/projects/nar/nar-validation/activations/llama31_8b/e11_calibration/factors/nar_b128_k8/down_layer_00.pt')
        if not path.is_file():self.skipTest('Local trained E14 factor is unavailable')
        factor=RotationFactor.load(path,torch.device('cpu'));w,y=compact_wy(factor.reflectors,factor.active)
        generator=torch.Generator().manual_seed(11)
        value=torch.randn(3,factor.n,generator=generator)
        signs=torch.randint(0,2,(factor.n,),generator=generator).float()*2-1
        original=WYFactor(factor,w,y)
        fast=FastOriginalE14R4(original,signs,hadamard_fn=dense_hadamard)
        self.assertEqual(factor.b,128)
        self.assertTrue(torch.allclose(fast(value),original.apply(value,signs),atol=3e-6,rtol=3e-6))


if __name__=='__main__':unittest.main()

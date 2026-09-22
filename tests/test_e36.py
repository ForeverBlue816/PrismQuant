import unittest
import torch
from nar import e36_offset_precision as e

class OffsetPrecisionContract(unittest.TestCase):
    def test_main_rounding_and_degenerate_groups(self):
        result=e.fixed_tensor()
        self.assertTrue(result['rounding_matches_main'])
    def test_precision_changes_only_reconstruction_metadata(self):
        x=(torch.arange(128).float()*.0137+19.12345).reshape(1,1,128)
        _,p,_=e.payload(x)
        for row in e.ROWS:
            s,z=e.metadata(p,row)
            if row!='S32':self.assertTrue(torch.equal(s,p['scale16']))
            if row in ['Z16','S32']:self.assertTrue(torch.equal(z,p['zero16']))
        self.assertEqual(e.metadata(p,'Z32')[1].dtype,torch.float32)
        self.assertEqual(e.metadata(p,'ZB16')[1].dtype,torch.bfloat16)
        self.assertEqual(e.metadata(p,'S32')[0].dtype,torch.float32)
    def test_frozen_payload_does_not_reencode_changed_upstream_input(self):
        class Identity:
            def apply(self,s,l,x,transpose=False):return x
        x=(torch.arange(128).float()*.0037+1.1256).reshape(1,1,128)
        _,p,_=e.payload(x)
        hook=e.Replay(None,Identity(),{('qkv',0):p},'Z32')
        first=hook.transform('qkv',0,x.bfloat16())
        second=hook.transform('qkv',0,(x+100).bfloat16())
        self.assertTrue(torch.equal(first,second))
    def test_hash_gate_detects_modified_codes_or_scale(self):
        _,p,_=e.payload(torch.linspace(-1,2,128).reshape(1,1,128))
        rows=[dict(precision=row,site='qkv',layer=0,**e.hashes(p,row)) for row in e.ROWS]
        self.assertTrue(e.verify_payload_hashes(rows))
        broken=[dict(r) for r in rows];broken[1]['code_sha256']='changed'
        with self.assertRaises(AssertionError):e.verify_payload_hashes(broken)
        broken=[dict(r) for r in rows];broken[2]['scale_sha256']='changed'
        with self.assertRaises(AssertionError):e.verify_payload_hashes(broken)
    def test_full_grid_is_fixed(self):
        self.assertEqual(len(e.plan()),8)
        self.assertEqual({r['row'] for r in e.plan()},{method+'_'+row for method in e.METHODS for row in e.ROWS})

if __name__=='__main__':unittest.main()

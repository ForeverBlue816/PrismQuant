import unittest
import numpy as np
from nar.e36_analysis import exact_quantiles,divide,measure,exact_stream_median

class ExactGroupStatistics(unittest.TestCase):
    def test_exact_pooled_quantile_retains_tail_and_infinity(self):
        q,info=exact_quantiles(np.array([0.,1.,2.,3.,np.inf,np.nan]))
        self.assertEqual(q,[2.,np.inf]);self.assertEqual(info['undefined_count'],1);self.assertEqual(info['infinite_count'],1)
    def test_streamed_radix_median_is_exact_with_ties_and_infinity(self):
        for values in [[0.,0.,1.,2.,np.inf],[0.,1.,1.,2.],[1e-250,1e-20,1.,2.,1e250,1e300],[np.inf,np.inf]]:
            x=np.array(values,dtype=np.float64)
            expected=exact_quantiles(x,(.5,))[0][0]
            self.assertEqual(exact_stream_median(x,block_size=2),expected)
    def test_zero_denominators_are_not_epsilon_clipped(self):
        out=divide(np.array([1.,0.,0.]),np.array([0.,0.,2.]))
        self.assertTrue(np.isinf(out[0]));self.assertTrue(np.isnan(out[1]));self.assertEqual(out[2],0.)
    def test_ratio_is_computed_per_group_before_median(self):
        x=np.array([[1.0001,1.,.1,.5],[2.0002,1.,.1,100.],[3.0003,1.,.1,.001]],dtype=np.float32)
        values=measure(x)
        np.testing.assert_allclose(values['error_to_unaligned'],values['offset_error_steps']/values['unaligned_step_steps'])
        self.assertNotAlmostEqual(float(np.median(values['error_to_unaligned'])),float(np.median(values['offset_error_steps'])/np.median(values['unaligned_step_steps'])),places=6)

if __name__=='__main__':unittest.main()

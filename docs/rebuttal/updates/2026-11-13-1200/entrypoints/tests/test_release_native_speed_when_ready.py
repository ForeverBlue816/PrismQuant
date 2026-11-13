"""Small CPU artifact fixtures; scheduler mutations are mocked."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1]/'scripts/release_native_speed_when_ready.py'
spec = importlib.util.spec_from_file_location('native_speed_release', SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ArtifactReadiness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.flat = Path(self.temp.name)/'flat'
        self.du = Path(self.temp.name)/'du'
        self.flat.mkdir(); self.du.mkdir()
        for directory, filenames in [(self.flat, ('quantized_model.pt', 'weight_quantizers.pt')),
                                     (self.du, ('quantized_model.pt', 'Rot.pkl', 'act_scales.pt', 'calibration.pt'))]:
            for name in filenames:
                (directory/name).write_bytes(b'fixture')
        self.flat_data = dict(status='COMPLETE', model=helper.MODEL, model_key='llama31_8b',
            base_checkpoint=str(helper.BASE), official_commit='9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1',
            official_args=dict(model=str(helper.BASE), seed=0, w_bits=4, a_bits=4, k_bits=4, v_bits=4, gptq=False),
            quantized_model=str(self.flat/'quantized_model.pt'), checkpoint_bytes=7)
        def identity(name):
            return dict(path=str(self.du/name), bytes=7, sha256='a'*64)
        self.du_data = dict(status='RUNNING', phase='quantize-ppl', seed=0, model=helper.MODEL,
            model_key='llama31_8b', base_revision=helper.BASE.name,
            official_commit='d56cfc6fe97c34c0eb100fec82fe439865905679',
            official_arguments=dict(model=str(helper.BASE), net=helper.MODEL, wbits=4, abits=4, quant_method='duquant'),
            quantization_seconds=1, checkpoint_save_seconds=1, checkpoint=identity('quantized_model.pt'),
            rotation_table=identity('Rot.pkl'), smoothing=dict(scales=identity('act_scales.pt')),
            calibration=dict(artifact=identity('calibration.pt')))
        self.save()

    def save(self):
        (self.flat/'DONE.json').write_text(json.dumps(self.flat_data))
        (self.du/'quantization_manifest.json').write_text(json.dumps(self.du_data))

    def test_complete_artifacts_accept_du_running_without_any_metric(self):
        result = helper.readiness(self.flat, self.du)
        self.assertTrue(result['ready'])
        self.assertEqual(result['artifacts']['duquant']['manifest_run_status'], 'RUNNING')

    def test_missing_manifest_does_not_call_scheduler(self):
        (self.flat/'DONE.json').unlink()
        result = helper.readiness(self.flat, self.du)
        runner = Mock()
        helper.release(result, apply=True, caller_job=helper.PARENT, run=runner)
        self.assertEqual(result['action'], 'NOT_READY')
        runner.assert_not_called()

    def test_truncated_or_missing_checkpoint_rejected(self):
        (self.du/'quantized_model.pt').write_bytes(b'bad')
        self.assertFalse(helper.readiness(self.flat, self.du)['ready'])
        (self.du/'quantized_model.pt').write_bytes(b'fixture')
        (self.flat/'weight_quantizers.pt').unlink()
        self.assertFalse(helper.readiness(self.flat, self.du)['ready'])

    def test_wrong_model_or_checkpoint_path_rejected(self):
        self.flat_data['model_key'] = 'llama32_3b'; self.save()
        self.assertFalse(helper.readiness(self.flat, self.du)['ready'])
        self.flat_data['model_key'] = 'llama31_8b'
        self.du_data['checkpoint']['path'] = str(self.flat/'quantized_model.pt'); self.save()
        self.assertFalse(helper.readiness(self.flat, self.du)['ready'])

    def test_exact_pending_job_released_once_then_changed_dependency_skipped(self):
        job = (f'JobId={helper.JOB} JobName=pq8b-official-speed UserId=user({os.getuid()}) '
               f'JobState=PENDING Dependency=afterany:{helper.PARENT}(unfulfilled) '
               f'Command={helper.ROOT}/slurm_rebuttal_speed_8b.sh')
        runner = Mock(side_effect=[subprocess.CompletedProcess([], 0, job, ''),
                                   subprocess.CompletedProcess([], 0, '', '')])
        result = helper.readiness(self.flat, self.du)
        helper.release(result, apply=True, caller_job=helper.PARENT, run=runner)
        self.assertEqual(result['action'], 'RELEASED')
        self.assertEqual(runner.call_args_list[1].args[0], ['scontrol', 'update', f'JobId={helper.JOB}', 'Dependency='])
        for dependency in ['(null)', 'afterany:999', f'afterany:{helper.PARENT}(unfulfilled),afterok:888']:
            with self.subTest(dependency=dependency):
                changed = job.replace(f'afterany:{helper.PARENT}(unfulfilled)', dependency)
                runner = Mock(return_value=subprocess.CompletedProcess([], 0, changed, ''))
                helper.release(result, apply=True, caller_job=helper.PARENT, run=runner)
                self.assertEqual(result['action'], 'SKIPPED')
                self.assertEqual(runner.call_count, 1)

    def test_dry_run_and_wrong_producer_never_call_scheduler(self):
        runner = Mock()
        result = helper.readiness(self.flat, self.du)
        helper.release(result, run=runner)
        self.assertEqual(result['action'], 'READY_DRY_RUN')
        helper.release(result, apply=True, caller_job='999', run=runner)
        self.assertEqual(result['action'], 'SKIPPED')
        runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()

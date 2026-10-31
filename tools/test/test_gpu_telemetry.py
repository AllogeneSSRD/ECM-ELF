"""Short measurements must expose samples before exit and report sampling failures."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
import gpu_telemetry


class FakeMonitor:
    def __init__(self,device):
        self.identity={'index':device};self.closed=False

    def sample(self):
        return {'gpu_percent':73,'sm_clock_mhz':1500}

    def close(self):
        self.closed=True


class TelemetryTest(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/'data/experiments/gpu_telemetry_unit'/uuid.uuid4().hex
        self.folder.mkdir(parents=True)

    @patch.object(gpu_telemetry,'Monitor',FakeMonitor)
    def test_short_measurement_flushes_before_close(self):
        path=self.folder/'telemetry.jsonl'
        recorder=gpu_telemetry.TelemetryRecorder(1,path,interval=60)
        try:
            deadline=time.monotonic()+2
            while not recorder.samples and time.monotonic()<deadline:
                time.sleep(.001)
            # The interval is still pending: an active, very short benchmark
            # already has a readable line instead of a buffered/empty file.
            self.assertEqual(json.loads(path.read_text())['sm_clock_mhz'],1500)
        finally:
            recorder.close()
        self.assertTrue(recorder.monitor.closed)
        self.assertEqual(len(path.read_text().splitlines()),recorder.samples)
        self.assertFalse(recorder.thread.is_alive())

    @patch.object(gpu_telemetry,'Monitor',FakeMonitor)
    def test_sampler_failure_is_not_a_successful_trace(self):
        with patch.object(FakeMonitor,'sample',side_effect=RuntimeError('probe failed')):
            recorder=gpu_telemetry.TelemetryRecorder(1,self.folder/'failed.jsonl')
            deadline=time.monotonic()+2
            while not recorder.errors and time.monotonic()<deadline:
                time.sleep(.001)
            with self.assertRaisesRegex(RuntimeError,'probe failed'):
                recorder.close()
            self.assertTrue(recorder.monitor.closed)


if __name__=='__main__':
    unittest.main()

"""Flush each read-only NVML observation; stopping never drops a buffered tail."""
import json
import threading
import time
from bench_stage2_n_scaling import Monitor


class TelemetryRecorder:
    def __init__(self, device, path, interval=.5):
        if interval <= 0:
            raise ValueError('telemetry interval must be positive')
        self.monitor = Monitor(device)
        self.identity = self.monitor.identity
        self.output = path.open('x', encoding='utf-8', buffering=1)
        self.interval = interval
        self.stop = threading.Event()
        self.errors = []
        self.samples = 0
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()

    def _record(self):
        try:
            while True:
                row = self.monitor.sample()
                row['elapsed_seconds'] = time.perf_counter() - self.started
                self.output.write(json.dumps(row) + '\n')
                self.output.flush()
                self.samples += 1
                if self.stop.wait(self.interval):
                    break
        except Exception as error:
            self.errors.append(str(error))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('NVML sampling did not stop; partial evidence retained')
        self.output.close()
        self.monitor.close()
        if self.errors:
            raise RuntimeError('NVML sampling failed: ' + '; '.join(self.errors))

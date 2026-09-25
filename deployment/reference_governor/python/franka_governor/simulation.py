"""CPU correctness adapter. Independent of Isaac/Torch; per-environment sessions."""
import numpy as np
from . import Batch, Feedback, Message, Status


class SimulationBatch:
    def __init__(self, config, count):
        self.count = count
        self.batch = Batch(config, count)
        self.sessions = np.zeros(count, dtype=np.uint64)
        self.sequences = np.zeros(count, dtype=np.uint64)
        self.initialized = np.zeros(count, dtype=bool)
        self.outputs = [self.batch.output(i) for i in range(count)]

    def _matrix(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.count, 7):
            raise ValueError(f"expected {(self.count, 7)}, got {x.shape}")
        return x

    def reset(self, ids, q, dq, now_ns):
        q, dq = self._matrix(q), self._matrix(dq)
        ids = list(ids)
        if len(set(ids)) != len(ids) or any(i < 0 or i >= self.count for i in ids):
            raise ValueError("reset indices must be unique and in range")
        for i in ids:
            self.sessions[i] += 1
            self.sequences[i] = 0
            f = Feedback()
            f.session, f.observed_ns = int(self.sessions[i]), now_ns
            f.q = f.desired_q = q[i].tolist()
            f.dq = f.desired_dq = dq[i].tolist()
            f.desired_ddq = [0.]*7  # simulator reset supplies no reference acceleration
            if not self.batch.reset(i, int(self.sessions[i]), now_ns, f):
                self.initialized[i] = False
                raise ValueError(f"infeasible reset in environment {i}")
            self.initialized[i] = True
            self.outputs[i] = self.batch.output(i)

    def submit(self, actions, observation_ns, completed_ns, now_ns):
        actions = self._matrix(actions).astype(np.float32)
        accepted = np.zeros(self.count, dtype=bool)
        for i in range(self.count):
            if not self.initialized[i]:
                raise RuntimeError("reset every environment before submission")
            if self.outputs[i].status != Status.RUNNING:
                continue
            m = Message()
            m.session, m.sequence = int(self.sessions[i]), int(self.sequences[i])+1
            m.action = actions[i].tolist()
            m.observation_ns, m.completed_ns = observation_ns, completed_ns
            accepted[i] = self.batch.submit(i, m, now_ns)
            if accepted[i]: self.sequences[i] += 1
            self.outputs[i] = self.batch.output(i)
        return accepted

    def step(self, now_ns, q, dq):
        if not self.initialized.all():
            raise RuntimeError("reset every environment before stepping")
        q, dq = self._matrix(q), self._matrix(dq)
        feedback = []
        for i, o in enumerate(self.outputs):
            f = Feedback()
            f.session, f.observed_ns = int(self.sessions[i]), now_ns
            f.q, f.dq = q[i].tolist(), dq[i].tolist()
            f.desired_q, f.desired_dq, f.desired_ddq = o.q, o.dq, o.ddq
            feedback.append(f)
        self.outputs = self.batch.step(now_ns, feedback)
        return self.positions

    @property
    def positions(self):
        return np.asarray([o.q for o in self.outputs], dtype=np.float64)

    @property
    def failed(self):
        return np.asarray([o.status in (Status.STOPPING, Status.TERMINAL, Status.FAULT)
                           for o in self.outputs], dtype=bool)

    @property
    def previous_raw(self):
        return np.asarray([o.previous_raw for o in self.outputs], dtype=np.float32)

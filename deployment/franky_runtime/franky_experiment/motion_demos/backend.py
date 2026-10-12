"""Transfer adapters. Importing this module never connects to FCI."""
import importlib.metadata


class FakeMotionBackend:
    """Reference/lifecycle recorder, deliberately not a robot dynamics emulator."""
    def __init__(self):self.events=[];self.active=False
    def start(self,task):
        if self.active:raise RuntimeError('Previous trial still active')
        self.active=True;self.events.append(('start',task['name']))
    def set_reference(self,sequence,reference):
        if not self.active:raise RuntimeError('No active trial')
        self.events.append(('reference',sequence,list(reference)))
    def stop(self):
        if self.active:self.events.append(('stop',));self.active=False
    def close(self):self.stop()


def audit_motion_capabilities():
    try:version=importlib.metadata.version('franky')
    except importlib.metadata.PackageNotFoundError:version=None
    return {'franky_version':version,'hardware_ready':False,
        'missing':['audited native Cartesian SO3 torque tracker and timestamped pose/twist publication',
            'reset-anchored dynamic torque projector using hardware model inertia/Jacobian',
            'M3 explicit posture targets with original kinematic projector',
            'native 1 kHz line/circle projection, normal impedance and tangent damping',
            'encoder-derived flange pose/twist; native buffered telemetry and applied-reference sequence',
            'workspace, singularity, joint/contact/velocity guards; watchdog and audited torque stop',
            'gain/torque filter/slew/FCI compensation parity and locally approved commissioning']}


class FrankyMotionBackend:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('Hardware mode unavailable: '+ '; '.join(audit_motion_capabilities()['missing']))

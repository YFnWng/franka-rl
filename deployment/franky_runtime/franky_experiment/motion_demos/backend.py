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
    try:
        version=importlib.metadata.version('franky-control')
        import franky
        native=hasattr(franky, 'M4RuntimeMotion')
    except (importlib.metadata.PackageNotFoundError, ImportError):
        version=None; native=False
    return {'franky_version':version, 'native_runtime_available':native,
            'hardware_ready':False,
            'missing': ([] if native else ['installed M4 native runtime wheel']) +
                       ['local reviewed commissioning configuration and hardware commissioning evidence'],
            'note':'Availability is distinct from physical commissioning.'}


from .hardware import FrankyMotionBackend

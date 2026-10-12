"""Offline fake/no-motion transfer validation. Not a Python torque servo."""
from pathlib import Path
import json
import numpy as np
from franka_motion.transfer import load_transfer_suite,prepare_task
from .backend import FakeMotionBackend,FrankyMotionBackend,audit_motion_capabilities


class FrankyMotionRunner:
    def __init__(self,bundle,suite):
        self.bundle=Path(bundle);self.suite=load_transfer_suite(suite)

    def run_suite(self,output):
        # Fail before creating artifacts or accessing a robot for unsupported hardware mode.
        if self.suite['mode']=='hardware':FrankyMotionBackend()
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        backend=FakeMotionBackend();results=[]
        try:
            for task in self.suite['tasks']:
                metadata,arrays=prepare_task(task,self.bundle/'config',self.bundle/'model/robot_returned_model.urdf')
                with np.load(self.bundle/'fixtures'/(task['name']+'.npz'),allow_pickle=False) as golden:
                    if set(golden.files)!=set(arrays):raise ValueError('Golden fixture fields mismatch')
                    for key,value in arrays.items():
                        if value.dtype.kind in 'fci':np.testing.assert_allclose(value,golden[key],atol=1e-9,rtol=1e-9)
                        else:np.testing.assert_array_equal(value,golden[key])
                backend.start(task)
                if self.suite['mode']=='fake' and task['kind']=='tracking':
                    for i,(p,q,v,w) in enumerate(zip(arrays['position_m'],arrays['quaternion_xyzw'],arrays['linear_velocity_m_s'],arrays['angular_velocity_rad_s'])):
                        backend.set_reference(i,np.r_[p,q,v,w])
                elif self.suite['mode']=='fake' and task['kind']=='posture':
                    for i,reference in enumerate(arrays['reference']):backend.set_reference(i,reference)
                backend.stop()
                np.savez_compressed(output/(task['name']+'.npz'),**arrays)
                results.append({'task':task['name'],'status':'offline_contract_checked','golden_parity_passed':True,'metadata':metadata})
        finally:backend.close()
        report={'mode':self.suite['mode'],'hardware_executed':False,'physical_acceptance':None,
            'limitations':'Fake records references only; no fabricated measured state, dynamics, hand force or hardware success.',
            'capabilities':audit_motion_capabilities(),'results':results,'lifecycle_events':backend.events}
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report

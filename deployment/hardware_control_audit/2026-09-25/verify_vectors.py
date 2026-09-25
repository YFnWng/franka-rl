"""Portable offline math verification. No ROS/Isaac/robot connection."""
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def velocity_limits(q, joints):
    upper, lower = [], []
    for i, x in enumerate(q, 1):
        j = joints[f'joint{i}']; lim = j['limit']; p = j['position_based_velocity_limits']
        vmax, off, dec = lim['velocity'], p['velocity_offset'], p['deceleration_limit']
        upper.append(min(vmax, max(0., -off + math.sqrt(max(0., 2*dec*(lim['upper']-x))))) - .001)
        lower.append(max(-vmax, min(0., off - math.sqrt(max(0., 2*dec*(x-lim['lower']))))) + .001)
    return np.array(upper), np.array(lower)

def limiter_step(target, q, v, a, upper, lower):
    # libfranka 0.19.0 scalar position limitRate equations, fixed 1 ms.
    # This helper is NOT a position/workspace/stale-command safety governor.
    dt, amax, jmax = .001, 9.999, 4999.999
    jerk = (((target-q)/dt-v)/dt-a)/dt
    desired_a = a + np.clip(jerk, -jmax, jmax)*dt
    upper_a = np.minimum(jmax/amax*(upper-v), amax)
    lower_a = np.maximum(jmax/amax*(lower-v), -amax)
    new_v = v + np.maximum(np.minimum(desired_a, upper_a), lower_a)*dt
    return q + new_v*dt

def verify():
    inputs = json.loads((ROOT/'inputs.json').read_text())
    expected = json.loads((ROOT/'vectors.json').read_text())
    joints = json.loads((ROOT/'limits.json').read_text())['joints']
    for c, e in zip(inputs['limiter_cases'], expected['limiter_cases'], strict=True):
        q, v, a, target = (np.array(c[k]) for k in ['q_d','dq_d','ddq_d','target'])
        upper, lower = velocity_limits(q, joints)
        np.testing.assert_allclose(upper,e['description_upper'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(lower,e['description_lower'],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(target,e['stock_default_output'])
        np.testing.assert_allclose(limiter_step(target,q,v,a,upper,lower),e['optional_description_limited'],rtol=0,atol=1e-12)
        gain=.001/(.001+1/(2*math.pi*100.))
        filtered=gain*target+(1-gain)*q
        np.testing.assert_allclose(filtered,e['optional_filter_100Hz'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(limiter_step(filtered,q,v,a,np.array(e['legacy_upper']),np.array(e['legacy_lower'])),e['optional_stock_filter_then_legacy_limit'],rtol=0,atol=1e-12)
    for c,e in zip(inputs['observation_cases'],expected['observation_cases'],strict=True):
        f=lambda k: np.asarray(c[k],dtype=np.float32)
        obs=np.concatenate([f('q')-f('default'),f('dq'),f('target_xyz')-f('flange_xyz'),f('previous_raw')])
        mapped=f('default')+np.float32(.5)*f('action')
        np.testing.assert_array_equal(obs,np.asarray(e['observation'],dtype=np.float32))
        np.testing.assert_array_equal(mapped,np.asarray(e['mapped_target'],dtype=np.float32))
        if 'expected_sentinel' in c:
            np.testing.assert_allclose(obs,c['expected_sentinel'],rtol=0,atol=3e-7)
    result={'limiter_cases':len(inputs['limiter_cases']),'observation_cases':len(inputs['observation_cases']),
            'observation_action_cpp_python':'float32 exact match','limiter_absolute_tolerance':1e-12,
            'hardware_accessed':False,'closed_loop_governor_verified':False}
    print(json.dumps(result,indent=2))

if __name__=='__main__': verify()

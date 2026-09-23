"""Pick-and-place using FrankaPickPlace."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils

app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")

from isaacsim.core.experimental.objects import DomeLight, GroundPlane
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.manipulators.examples.franka import FrankaPickPlace

DEVICE = "cpu"

GroundPlane("/World/ground_plane")
dome_light = DomeLight("/World/DomeLight")
dome_light.set_intensities(1000)

# -- Begin custom setup -- #
# Custom phase durations (steps for each phase)
controller = FrankaPickPlace(events_dt=[80, 60, 30, 60, 100, 30, 30])
# Customize cube position, size, and target position
controller.setup_scene(
    cube_initial_position=[0.4, 0.2, 0.0258], cube_size=[0.05, 0.05, 0.05], target_position=[-0.4, 0.2, 0.12]
)
# -- End of custom setup -- #

SimulationManager.setup_simulation(dt=1.0 / 60.0, device=DEVICE)
physics_scene = SimulationManager.get_physics_scenes()[0]
physics_scene.set_enabled_gpu_dynamics(False)
app_utils.play()
# Run a few steps so the articulation's physics tensor entity is valid before `controller.reset()`
app_utils.update_app(steps=20)
controller.reset()

# Main loop: run one pick-place step each physics frame until done
step_count = 0
max_test_steps = sum(controller.events_dt) + 60
while simulation_app.is_running():
    simulation_app.update()
    step_count += 1
    if app_utils.is_playing():
        if not controller.is_done():
            controller.forward()
        else:
            print("Pick-and-place completed")
            app_utils.pause()
            if args.test:
                break
    if args.test and step_count >= max_test_steps:
        raise RuntimeError("Pick-and-place did not complete within the test step budget")

app_utils.stop()
simulation_app.close()
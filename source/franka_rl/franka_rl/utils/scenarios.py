"""Named scenario loading and Isaac Lab environment modification."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg


ARM_JOINT_PATTERN = "panda_joint.*"
ARM_BODY_PATTERN = "panda_link[1-7]|panda_hand"
ARM_ACTUATOR_NAMES = ("panda_shoulder", "panda_forearm")
DEFAULT_SCENARIO_RESOURCE = "config/scenarios.yaml"

ScenarioType = Literal["nominal", "specified", "random"]
DistributionName = Literal["uniform", "log_uniform", "gaussian"]


@dataclass(frozen=True)
class DistributionSpec:
    """Two-parameter distribution accepted by Isaac Lab randomizers."""

    distribution: DistributionName
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.distribution in ("uniform", "log_uniform") and self.low > self.high:
            raise ValueError(f"Invalid {self.distribution} range [{self.low}, {self.high}].")
        if self.distribution == "gaussian" and self.high <= 0.0:
            raise ValueError("For a gaussian distribution, 'low' is the mean and 'high' must be a positive std.")

    @property
    def parameters(self) -> tuple[float, float]:
        return (self.low, self.high)


@dataclass(frozen=True)
class PhysicsSpec:
    stiffness_scale: float | DistributionSpec | None = None
    damping_scale: float | DistributionSpec | None = None
    effort_limit_scale: float | DistributionSpec | None = None
    joint_friction_add: float | DistributionSpec | None = None
    joint_armature_scale: float | DistributionSpec | None = None
    link_mass_scale: float | DistributionSpec | None = None
    link_inertia_scale: float | DistributionSpec | None = None


@dataclass(frozen=True)
class ObservationSpec:
    joint_position_noise: DistributionSpec | None = None
    joint_velocity_noise: DistributionSpec | None = None
    ee_position_error_noise: DistributionSpec | None = None
    joint_position_bias: DistributionSpec | None = None
    joint_velocity_bias: DistributionSpec | None = None
    ee_position_error_bias: DistributionSpec | None = None


@dataclass(frozen=True)
class ControlSpec:
    """Controller-interface perturbations applied outside the scene config."""

    action_delay_steps: int | None = None


@dataclass(frozen=True)
class ResetSpec:
    """Episode initialization variations."""

    joint_position_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    type: ScenarioType
    description: str = ""
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    observations: ObservationSpec = field(default_factory=ObservationSpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    reset: ResetSpec = field(default_factory=ResetSpec)
    resampling: Literal["startup", "reset"] | None = None

    @property
    def records_episode_parameters(self) -> bool:
        return self.type in ("nominal", "specified")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScenarioCatalog:
    """Strict loader for named scenario definitions."""

    def __init__(self, scenarios: dict[str, ScenarioSpec], *, source: str, sha256: str, version: int):
        self._scenarios = scenarios
        self.source = source
        self.sha256 = sha256
        self.version = version

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> ScenarioCatalog:
        if path is None:
            resource = files("franka_rl").joinpath(DEFAULT_SCENARIO_RESOURCE)
            raw = resource.read_bytes()
            source = f"package:{DEFAULT_SCENARIO_RESOURCE}"
        else:
            scenario_path = Path(path).expanduser().resolve()
            if not scenario_path.is_file():
                raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
            raw = scenario_path.read_bytes()
            source = str(scenario_path)

        document = yaml.safe_load(raw) or {}
        _require_mapping(document, "scenario document")
        _reject_unknown_keys(document, {"version", "scenarios"}, "scenario document")
        version = document.get("version")
        if version != 1:
            raise ValueError(f"Unsupported scenario schema version {version!r}; expected 1.")

        raw_scenarios = _require_mapping(document.get("scenarios"), "scenarios")
        scenarios = {
            name: _parse_scenario(name, value)
            for name, value in raw_scenarios.items()
        }
        if not scenarios:
            raise ValueError("Scenario catalog must define at least one scenario.")
        return cls(
            scenarios,
            source=source,
            sha256=hashlib.sha256(raw).hexdigest(),
            version=version,
        )

    def get(self, name: str) -> ScenarioSpec:
        try:
            return self._scenarios[name]
        except KeyError as error:
            available = ", ".join(sorted(self._scenarios))
            raise ValueError(f"Unknown scenario {name!r}. Available scenarios: {available}.") from error

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._scenarios))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "source": self.source,
            "sha256": self.sha256,
        }


class ScenarioModifier:
    """Apply one resolved scenario using Isaac Lab configuration and events."""

    PARAMETER_SCHEMA = {
        "joint_stiffness": {"unit": "N*m/rad"},
        "joint_damping": {"unit": "N*m*s/rad"},
        "joint_effort_limit": {"unit": "N*m"},
        "joint_friction_coefficient": {"unit": "backend_specific"},
        "joint_armature": {"unit": "kg*m^2"},
        "link_mass": {"unit": "kg"},
        "link_inertia_diagonal": {"unit": "kg*m^2"},
    }

    def __init__(self, scenario: ScenarioSpec, catalog: ScenarioCatalog):
        self.spec = scenario
        self.catalog = catalog
        self._joint_ids: list[int] | None = None
        self._joint_names: list[str] | None = None
        self._configured_values: dict[str, Any] = {}

    @property
    def records_episode_parameters(self) -> bool:
        return self.spec.records_episode_parameters

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "catalog": self.catalog.metadata(),
            "configured_values": self._configured_values,
        }

    def apply(self, env_cfg) -> dict[str, Any]:
        """Modify ``env_cfg`` before construction and return resolved metadata."""

        if self.spec.type == "nominal":
            return self.metadata
        if self.spec.type == "specified":
            self._apply_specified_physics(env_cfg)
        elif self.spec.type == "random":
            self._apply_random_physics(env_cfg)
        else:  # pragma: no cover - guarded by YAML parsing
            raise ValueError(f"Unsupported scenario type: {self.spec.type}")

        self._apply_observation_noise(env_cfg)
        self._apply_reset_variation(env_cfg)
        return self.metadata

    def _apply_specified_physics(self, env_cfg) -> None:
        physics = self.spec.physics
        for field_name in (
            "stiffness_scale",
            "damping_scale",
            "effort_limit_scale",
            "joint_friction_add",
            "joint_armature_scale",
            "link_mass_scale",
            "link_inertia_scale",
        ):
            value = getattr(physics, field_name)
            if isinstance(value, DistributionSpec):
                raise TypeError(f"Specified scenario property {field_name!r} must be a scalar.")

        if physics.stiffness_scale is not None:
            self._scale_actuator_cfg(env_cfg, "stiffness", physics.stiffness_scale)
        if physics.damping_scale is not None:
            self._scale_actuator_cfg(env_cfg, "damping", physics.damping_scale)
        if physics.effort_limit_scale is not None:
            self._scale_actuator_cfg(env_cfg, "effort_limit_sim", physics.effort_limit_scale)

        if physics.joint_friction_add is not None:
            env_cfg.events.scenario_joint_friction = EventTermCfg(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[ARM_JOINT_PATTERN]),
                    "friction_distribution_params": (
                        physics.joint_friction_add,
                        physics.joint_friction_add,
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                },
            )
            self._configured_values["joint_friction_add"] = physics.joint_friction_add

        if physics.joint_armature_scale is not None:
            env_cfg.events.scenario_joint_armature = EventTermCfg(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[ARM_JOINT_PATTERN]),
                    "armature_distribution_params": (
                        physics.joint_armature_scale,
                        physics.joint_armature_scale,
                    ),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            )
            self._configured_values["joint_armature_scale"] = physics.joint_armature_scale

        if physics.link_mass_scale is not None:
            env_cfg.events.scenario_link_mass = EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=[ARM_BODY_PATTERN]),
                    "mass_distribution_params": (physics.link_mass_scale, physics.link_mass_scale),
                    "operation": "scale",
                    "distribution": "uniform",
                    "recompute_inertia": True,
                },
            )
            self._configured_values["link_mass_scale"] = physics.link_mass_scale

        if physics.link_inertia_scale is not None:
            env_cfg.events.scenario_link_inertia = EventTermCfg(
                func=mdp.randomize_rigid_body_inertia,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=[ARM_BODY_PATTERN]),
                    "inertia_distribution_params": (
                        physics.link_inertia_scale,
                        physics.link_inertia_scale,
                    ),
                    "operation": "scale",
                    "distribution": "uniform",
                    "diagonal_only": True,
                },
            )
            self._configured_values["link_inertia_scale"] = physics.link_inertia_scale

    def _scale_actuator_cfg(self, env_cfg, attribute: str, scale: float) -> None:
        values: dict[str, dict[str, float]] = {}
        for actuator_name in ARM_ACTUATOR_NAMES:
            actuator_cfg = env_cfg.scene.robot.actuators[actuator_name]
            nominal = getattr(actuator_cfg, attribute)
            if not isinstance(nominal, (int, float)) or isinstance(nominal, bool):
                raise TypeError(
                    f"Expected scalar {attribute} for actuator {actuator_name!r}, "
                    f"got {type(nominal).__name__}."
                )
            applied = float(nominal) * scale
            setattr(actuator_cfg, attribute, applied)
            values[actuator_name] = {"nominal": float(nominal), "applied": applied}
        self._configured_values[attribute] = values

    def _apply_random_physics(self, env_cfg) -> None:
        physics = self.spec.physics
        mode = self.spec.resampling
        if mode is None:
            raise ValueError(f"Random scenario {self.spec.name!r} requires a resampling mode.")
        asset_cfg = SceneEntityCfg("robot", joint_names=[ARM_JOINT_PATTERN])

        stiffness = physics.stiffness_scale
        damping = physics.damping_scale
        if stiffness is not None:
            stiffness = _require_distribution(stiffness, "stiffness_scale")
        if damping is not None:
            damping = _require_distribution(damping, "damping_scale")

        if stiffness is not None and damping is not None and stiffness.distribution == damping.distribution:
            env_cfg.events.scenario_actuator_gains = EventTermCfg(
                func=mdp.randomize_actuator_gains,
                mode=mode,
                params={
                    "asset_cfg": asset_cfg,
                    "stiffness_distribution_params": stiffness.parameters,
                    "damping_distribution_params": damping.parameters,
                    "operation": "scale",
                    "distribution": stiffness.distribution,
                },
            )
        else:
            if stiffness is not None:
                env_cfg.events.scenario_stiffness = EventTermCfg(
                    func=mdp.randomize_actuator_gains,
                    mode=mode,
                    params={
                        "asset_cfg": asset_cfg,
                        "stiffness_distribution_params": stiffness.parameters,
                        "operation": "scale",
                        "distribution": stiffness.distribution,
                    },
                )
            if damping is not None:
                env_cfg.events.scenario_damping = EventTermCfg(
                    func=mdp.randomize_actuator_gains,
                    mode=mode,
                    params={
                        "asset_cfg": asset_cfg,
                        "damping_distribution_params": damping.parameters,
                        "operation": "scale",
                        "distribution": damping.distribution,
                    },
                )

        if physics.effort_limit_scale is not None:
            effort = _require_distribution(physics.effort_limit_scale, "effort_limit_scale")
            env_cfg.events.scenario_effort_limits = EventTermCfg(
                func=RandomizeJointEffortLimits,
                mode=mode,
                params={
                    "asset_cfg": asset_cfg,
                    "distribution_params": effort.parameters,
                    "distribution": effort.distribution,
                },
            )

        if physics.joint_friction_add is not None:
            friction = _require_distribution(physics.joint_friction_add, "joint_friction_add")
            env_cfg.events.scenario_joint_friction = EventTermCfg(
                func=mdp.randomize_joint_parameters,
                mode=mode,
                params={
                    "asset_cfg": asset_cfg,
                    "friction_distribution_params": friction.parameters,
                    "operation": "add",
                    "distribution": friction.distribution,
                },
            )

        if physics.joint_armature_scale is not None:
            armature = _require_distribution(physics.joint_armature_scale, "joint_armature_scale")
            env_cfg.events.scenario_joint_armature = EventTermCfg(
                func=mdp.randomize_joint_parameters,
                mode=mode,
                params={
                    "asset_cfg": asset_cfg,
                    "armature_distribution_params": armature.parameters,
                    "operation": "scale",
                    "distribution": armature.distribution,
                },
            )

        body_cfg = SceneEntityCfg("robot", body_names=[ARM_BODY_PATTERN])
        if physics.link_mass_scale is not None:
            mass = _require_distribution(physics.link_mass_scale, "link_mass_scale")
            env_cfg.events.scenario_link_mass = EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode=mode,
                params={
                    "asset_cfg": body_cfg,
                    "mass_distribution_params": mass.parameters,
                    "operation": "scale",
                    "distribution": mass.distribution,
                    "recompute_inertia": True,
                },
            )

        if physics.link_inertia_scale is not None:
            inertia = _require_distribution(physics.link_inertia_scale, "link_inertia_scale")
            env_cfg.events.scenario_link_inertia = EventTermCfg(
                func=mdp.randomize_rigid_body_inertia,
                mode=mode,
                params={
                    "asset_cfg": body_cfg,
                    "inertia_distribution_params": inertia.parameters,
                    "operation": "scale",
                    "distribution": inertia.distribution,
                    "diagonal_only": True,
                },
            )

    def _apply_observation_noise(self, env_cfg) -> None:
        observations = self.spec.observations
        terms = (
            ("joint_position_noise", "joint_position_bias", "joint_pos_rel"),
            ("joint_velocity_noise", "joint_velocity_bias", "joint_vel_rel"),
            ("ee_position_error_noise", "ee_position_error_bias", "ee_position_error"),
        )
        enabled = False
        for noise_name, bias_name, term_name in terms:
            noise = getattr(observations, noise_name)
            bias = getattr(observations, bias_name)
            if noise is None and bias is None:
                continue
            for field_name, distribution in ((noise_name, noise), (bias_name, bias)):
                if distribution is not None and distribution.distribution != "uniform":
                    raise ValueError(
                        f"Observation property {field_name!r} currently supports only uniform noise."
                    )
            observation_term = getattr(env_cfg.observations.policy, term_name)
            if bias is None:
                observation_term.noise = UniformNoiseCfg(n_min=noise.low, n_max=noise.high)
            else:
                per_step_noise = (
                    UniformNoiseCfg(n_min=noise.low, n_max=noise.high)
                    if noise is not None
                    else ConstantNoiseCfg(bias=0.0)
                )
                observation_term.noise = NoiseModelWithAdditiveBiasCfg(
                    noise_cfg=per_step_noise,
                    bias_noise_cfg=UniformNoiseCfg(n_min=bias.low, n_max=bias.high),
                )
            enabled = True
        if enabled:
            env_cfg.observations.policy.enable_corruption = True

    def _apply_reset_variation(self, env_cfg) -> None:
        joint_range = self.spec.reset.joint_position_range
        if joint_range is None:
            return
        env_cfg.events.reset_arm.params["position_range"] = joint_range
        self._configured_values["reset_joint_position_range"] = joint_range

    def capture(self, base_env) -> dict[str, torch.Tensor]:
        """Read realized arm parameters, one row per environment."""

        robot = base_env.scene["robot"]
        if self._joint_ids is None:
            self._joint_ids, self._joint_names = robot.find_joints(ARM_JOINT_PATTERN)
            self._body_ids, self._body_names = robot.find_bodies(ARM_BODY_PATTERN)
        joint_ids = self._joint_ids
        body_ids = self._body_ids
        inertia_diagonal = robot.data.body_inertia.torch[:, body_ids][..., (0, 4, 8)]
        return {
            "joint_stiffness": robot.data.joint_stiffness.torch[:, joint_ids].clone(),
            "joint_damping": robot.data.joint_damping.torch[:, joint_ids].clone(),
            "joint_effort_limit": robot.data.joint_effort_limits.torch[:, joint_ids].clone(),
            "joint_friction_coefficient": robot.data.joint_friction_coeff.torch[:, joint_ids].clone(),
            "joint_armature": robot.data.joint_armature.torch[:, joint_ids].clone(),
            "link_mass": robot.data.body_mass.torch[:, body_ids].clone(),
            "link_inertia_diagonal": inertia_diagonal.reshape(base_env.num_envs, -1).clone(),
        }

    @property
    def parameter_schema(self) -> dict[str, Any]:
        if self._joint_names is None:
            raise RuntimeError("capture() must be called before reading parameter_schema.")
        joint_names = list(self._joint_names)
        body_names = list(self._body_names)
        return {
            "joint_stiffness": {**self.PARAMETER_SCHEMA["joint_stiffness"], "component_names": joint_names},
            "joint_damping": {**self.PARAMETER_SCHEMA["joint_damping"], "component_names": joint_names},
            "joint_effort_limit": {**self.PARAMETER_SCHEMA["joint_effort_limit"], "component_names": joint_names},
            "joint_friction_coefficient": {
                **self.PARAMETER_SCHEMA["joint_friction_coefficient"],
                "component_names": joint_names,
            },
            "joint_armature": {**self.PARAMETER_SCHEMA["joint_armature"], "component_names": joint_names},
            "link_mass": {**self.PARAMETER_SCHEMA["link_mass"], "component_names": body_names},
            "link_inertia_diagonal": {
                **self.PARAMETER_SCHEMA["link_inertia_diagonal"],
                "component_names": [f"{body}:{axis}" for body in body_names for axis in ("Ixx", "Iyy", "Izz")],
            },
        }

    def validate_runtime(self, base_env, tolerance: float = 1.0e-5) -> dict[str, Any]:
        """Verify that fixed scenarios are identical across environments."""

        if not self.records_episode_parameters:
            return {"performed": False, "reason": "random scenario"}
        parameters = self.capture(base_env)
        maximum_spread = {
            name: float((values.max(dim=0).values - values.min(dim=0).values).abs().max())
            for name, values in parameters.items()
        }
        return {
            "performed": True,
            "passed": all(spread <= tolerance for spread in maximum_spread.values()),
            "tolerance": tolerance,
            "maximum_cross_environment_spread": maximum_spread,
        }


class RandomizeJointEffortLimits(ManagerTermBase):
    """Randomize physics effort limits relative to their initial values."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        self.default_limits = self.asset.data.joint_effort_limits.torch.clone()

    def __call__(
        self,
        env,
        env_ids,
        asset_cfg,
        distribution_params: tuple[float, float],
        distribution: DistributionName,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.asset.device, dtype=torch.int32)
        if isinstance(self.asset_cfg.joint_ids, slice):
            joint_ids = torch.arange(self.asset.num_joints, device=self.asset.device, dtype=torch.int32)
        else:
            joint_ids = torch.as_tensor(self.asset_cfg.joint_ids, device=self.asset.device, dtype=torch.int32)

        shape = (len(env_ids), len(joint_ids))
        if distribution == "uniform":
            scales = torch.empty(shape, device=self.asset.device).uniform_(*distribution_params)
        elif distribution == "log_uniform":
            low, high = distribution_params
            scales = torch.empty(shape, device=self.asset.device).uniform_(math.log(low), math.log(high)).exp_()
        elif distribution == "gaussian":
            mean, std = distribution_params
            scales = torch.empty(shape, device=self.asset.device).normal_(mean, std)
        else:  # pragma: no cover - guarded by YAML parsing
            raise ValueError(f"Unsupported distribution: {distribution}")
        if torch.any(scales < 0.0):
            raise ValueError("Effort-limit scales must be nonnegative.")

        limits = self.default_limits[env_ids[:, None], joint_ids] * scales
        self.asset.write_joint_effort_limit_to_sim_index(
            limits=limits,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )


def _parse_scenario(name: str, value: Any) -> ScenarioSpec:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError(
            f"Invalid scenario name {name!r}; use letters, numbers, '.', '_', and '-'."
        )
    mapping = _require_mapping(value, f"scenario {name!r}")
    _reject_unknown_keys(
        mapping,
        {"type", "description", "physics", "observations", "control", "reset", "resampling"},
        name,
    )
    scenario_type = mapping.get("type")
    if scenario_type not in ("nominal", "specified", "random"):
        raise ValueError(f"Scenario {name!r} has invalid type {scenario_type!r}.")

    physics = _parse_physics(mapping.get("physics"), scenario_type, name)
    observations = _parse_observations(mapping.get("observations"), name)
    control = _parse_control(mapping.get("control"), scenario_type, name)
    reset = _parse_reset(mapping.get("reset"), scenario_type, name)
    resampling = mapping.get("resampling")
    if resampling not in (None, "startup", "reset"):
        raise ValueError(f"Scenario {name!r} has invalid resampling mode {resampling!r}.")

    has_modifiers = (
        any(value is not None for value in asdict(physics).values())
        or any(value is not None for value in asdict(observations).values())
        or any(value is not None for value in asdict(control).values())
        or any(value is not None for value in asdict(reset).values())
    )
    if scenario_type == "nominal" and has_modifiers:
        raise ValueError(f"Nominal scenario {name!r} cannot define modifiers.")
    if scenario_type == "nominal" and resampling is not None:
        raise ValueError(f"Nominal scenario {name!r} cannot define resampling.")
    if scenario_type == "random" and resampling is None:
        raise ValueError(f"Random scenario {name!r} must define resampling.")
    if scenario_type == "specified" and resampling is not None:
        raise ValueError(f"Specified scenario {name!r} cannot define resampling.")

    return ScenarioSpec(
        name=name,
        type=scenario_type,
        description=str(mapping.get("description", "")),
        physics=physics,
        observations=observations,
        control=control,
        reset=reset,
        resampling=resampling,
    )


def _parse_physics(value: Any, scenario_type: ScenarioType, scenario_name: str) -> PhysicsSpec:
    if value is None:
        return PhysicsSpec()
    mapping = _require_mapping(value, f"physics for scenario {scenario_name!r}")
    allowed = {
        "stiffness_scale",
        "damping_scale",
        "effort_limit_scale",
        "joint_friction_add",
        "joint_armature_scale",
        "link_mass_scale",
        "link_inertia_scale",
    }
    _reject_unknown_keys(mapping, allowed, f"physics for scenario {scenario_name!r}")
    parsed: dict[str, Any] = {}
    for key, raw_value in mapping.items():
        if scenario_type == "specified":
            parsed[key] = _require_number(raw_value, f"{scenario_name}.{key}")
        elif scenario_type == "random":
            parsed[key] = _parse_distribution(raw_value, f"{scenario_name}.{key}")
        else:
            parsed[key] = raw_value
        _validate_physics_value(key, parsed[key], f"{scenario_name}.{key}")
    return PhysicsSpec(**parsed)


def _parse_observations(value: Any, scenario_name: str) -> ObservationSpec:
    if value is None:
        return ObservationSpec()
    mapping = _require_mapping(value, f"observations for scenario {scenario_name!r}")
    allowed = {
        "joint_position_noise",
        "joint_velocity_noise",
        "ee_position_error_noise",
        "joint_position_bias",
        "joint_velocity_bias",
        "ee_position_error_bias",
    }
    _reject_unknown_keys(mapping, allowed, f"observations for scenario {scenario_name!r}")
    return ObservationSpec(
        **{
            key: _parse_distribution(raw_value, f"{scenario_name}.{key}")
            for key, raw_value in mapping.items()
        }
    )


def _parse_control(value: Any, scenario_type: ScenarioType, scenario_name: str) -> ControlSpec:
    if value is None:
        return ControlSpec()
    mapping = _require_mapping(value, f"control for scenario {scenario_name!r}")
    _reject_unknown_keys(mapping, {"action_delay_steps"}, f"control for scenario {scenario_name!r}")
    delay = mapping.get("action_delay_steps")
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
        raise ValueError(f"{scenario_name}.action_delay_steps must be a nonnegative integer.")
    if scenario_type == "random":
        raise ValueError(
            f"Random scenario {scenario_name!r} cannot yet randomize action delay; "
            "use a fixed specified evaluation scenario."
        )
    return ControlSpec(action_delay_steps=delay)


def _parse_reset(value: Any, scenario_type: ScenarioType, scenario_name: str) -> ResetSpec:
    if value is None:
        return ResetSpec()
    mapping = _require_mapping(value, f"reset for scenario {scenario_name!r}")
    _reject_unknown_keys(mapping, {"joint_position_range"}, f"reset for scenario {scenario_name!r}")
    if scenario_type == "random":
        raise ValueError(
            f"Random scenario {scenario_name!r} cannot currently override reset ranges."
        )
    raw_range = mapping.get("joint_position_range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise TypeError(f"{scenario_name}.joint_position_range must be a two-element list.")
    low = _require_number(raw_range[0], f"{scenario_name}.joint_position_range[0]")
    high = _require_number(raw_range[1], f"{scenario_name}.joint_position_range[1]")
    if low > high:
        raise ValueError(f"{scenario_name}.joint_position_range lower bound exceeds upper bound.")
    return ResetSpec(joint_position_range=(low, high))


def _parse_distribution(value: Any, context: str) -> DistributionSpec:
    mapping = _require_mapping(value, context)
    _reject_unknown_keys(mapping, {"distribution", "low", "high"}, context)
    if set(mapping) != {"distribution", "low", "high"}:
        raise ValueError(f"{context} must define distribution, low, and high.")
    distribution = mapping["distribution"]
    if distribution not in ("uniform", "log_uniform", "gaussian"):
        raise ValueError(f"{context} has unsupported distribution {distribution!r}.")
    return DistributionSpec(
        distribution=distribution,
        low=_require_number(mapping["low"], f"{context}.low"),
        high=_require_number(mapping["high"], f"{context}.high"),
    )


def _require_distribution(value: float | DistributionSpec, context: str) -> DistributionSpec:
    if not isinstance(value, DistributionSpec):
        raise TypeError(f"Random scenario property {context!r} must be a distribution.")
    return value


def _validate_physics_value(key: str, value: float | DistributionSpec, context: str) -> None:
    if key == "joint_friction_add":
        return
    if isinstance(value, float):
        if value <= 0.0:
            raise ValueError(f"Scale {context} must be positive.")
        return
    if value.distribution == "gaussian":
        raise ValueError(
            f"Scale {context} cannot use an unbounded gaussian distribution because it can sample negatives."
        )
    if value.low <= 0.0:
        raise ValueError(f"Scale distribution {context} must have a positive lower bound.")


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping for {context}, got {type(value).__name__}.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"All keys in {context} must be strings.")
    return value


def _require_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Expected a number for {context}, got {type(value).__name__}.")
    return float(value)


def _reject_unknown_keys(mapping: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {', '.join(sorted(unknown))}.")

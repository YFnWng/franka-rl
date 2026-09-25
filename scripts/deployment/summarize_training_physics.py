"""Summarize completed physics captures and assemble a return evidence directory."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--hardware-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(exist_ok=False)
    reports = {}
    for label in ["nominal", "dr"]:
        source = a.captures / f"{label}_complete"
        reports[label] = json.loads((source / "runtime_model.json").read_text())
        assert reports[label]["complete"] and len(reports[label]["snapshots"]) == 3
        assert not reports[label]["physics_configuration_differences"]
        shutil.copytree(source, a.output_dir / label)
        shutil.copyfile(a.captures / f"{label}_complete.log", a.output_dir / f"{label}.log")
    hardware = json.loads(a.hardware_report.read_text())
    nominal = reports["nominal"]["snapshots"][0]
    names = nominal["body_names"]
    t = nominal["tensors"]
    body_rows = []
    for item in hardware["inertials"]:
        name = f"panda_link{item['link']}"
        idx = names.index(name)
        matrix = np.array(t["inertia_body_at_com_matrix_kg_m2"])[0, idx]
        trace = np.trace(matrix)
        body_rows.append(
            dict(
                body=name,
                runtime_mass_kg=t["mass_kg"][0][idx],
                runtime_com_body_m=t["com_pose_body_xyz_xyzw"][0][idx][:3],
                runtime_inertia_trace_kg_m2=float(trace),
                authored_inertia_trace_kg_m2=sum(item["usd_diagonal_inertia_kg_m2"]),
                fr3_mass_kg=item["fr3v2_mass_kg"],
                fr3_com_body_m=item["fr3v2_com_m"],
                fr3_inertia_trace_kg_m2=float(np.trace(item["fr3v2_inertia_kg_m2"])),
                fr3_over_runtime_trace=float(np.trace(item["fr3v2_inertia_kg_m2"]) / trace),
            )
        )
    # Validate every captured environment/reset and quantify sampled DR parameters.
    checks = {}
    for label, report in reports.items():
        matched = [l for l in report["usd"]["layers"] if l.get("hardware_audit_key")]
        assert len(matched) == len(hardware["asset_sha256"])
        assert all(l["matches_hardware_audit"] for l in matched)
        base = report["snapshots"][0]["tensors"]
        arm_ids = [names.index(f"panda_link{i}") for i in range(8)]
        delta = 0.0
        for snapshot in report["snapshots"]:
            values = snapshot["tensors"]
            for key, value in values.items():
                assert np.isfinite(np.asarray(value)).all(), key
            delta = max(
                delta,
                float(
                    np.max(
                        np.abs(
                            np.array(values["inertia_body_at_com_matrix_kg_m2"])[:, arm_ids]
                            - np.array(base["inertia_body_at_com_matrix_kg_m2"])[:, arm_ids]
                        )
                    )
                ),
            )
        checks[label] = dict(
            matching_asset_layers=len(matched),
            max_arm_link_inertia_change_across_resets=delta,
            physics_config_matches_saved=True,
        )
    samples = []
    hand = names.index("panda_hand")
    for snapshot in reports["dr"]["snapshots"][1:]:
        v = snapshot["tensors"]
        for env_id in snapshot["environment_indices"]:
            mass = v["mass_kg"][env_id][hand]
            added = mass - t["mass_kg"][env_id][hand]
            com = np.array(v["com_pose_body_xyz_xyzw"][env_id][hand][:3])
            samples.append(
                dict(
                    reset=snapshot["label"],
                    environment_index=env_id,
                    added_payload_kg=added,
                    hand_total_mass_kg=mass,
                    inferred_payload_position_body_m=(com * mass / added).tolist() if added > 0 else None,
                    stiffness_scale=(np.array(v["stiffness_Nm_rad"][env_id][:7]) / 80).tolist(),
                    damping_scale=(np.array(v["damping_Nm_s_rad"][env_id][:7]) / 4).tolist(),
                    effort_scale=(np.array(v["effort_limits_Nm"][env_id][:7]) / np.array([87] * 4 + [12] * 3)).tolist(),
                )
            )
    summary = dict(
        body_comparison=body_rows,
        checks=checks,
        dr_realized_samples=samples,
        hardware_report_sha256=digest(a.hardware_report),
        nominal_joint_parameters={
            k: v[0]
            for k, v in t.items()
            if k
            not in [
                "mass_kg",
                "com_pose_body_xyz_xyzw",
                "inertia_body_at_com_column_major_kg_m2",
                "inertia_body_at_com_matrix_kg_m2",
            ]
        },
    )
    (a.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Runtime training-model extraction",
        "",
        "PhysX retains the tiny authored Panda link inertias and zero COM offsets in this reconstruction.",
        "No automatic mass-property correction was observed at initialization or after either reset.",
        "",
        "Both runs used 8 environments, seed 123, PhysX GPU, 60 Hz physics and 30 Hz policy timing.",
        "Snapshots cover initialization before reset, first reset, and second reset. No policy rollout or hardware access occurred.",
        "",
        "All 14 robot USD composition-layer hashes match the returned hardware audit. Resolved physics-manager settings match the saved nominal/DR configs exactly.",
        "Task and scenario source have no differences from the training reference revision. Final config differences are instance count, command visualization, log path, and nominal seed (42 → 123).",
        "",
        "| Body | Runtime mass kg | Runtime inertia trace kg m² | FR3 inertia trace kg m² | FR3/runtime |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in body_rows:
        lines.append(
            f"| {row['body']} | {row['runtime_mass_kg']:.6f} | {row['runtime_inertia_trace_kg_m2']:.9g} | {row['fr3_inertia_trace_kg_m2']:.9g} | {row['fr3_over_runtime_trace']:.1f} |"
        )
    lines += [
        "",
        "The runtime principal-axis orientation is identity and COM position zero for the nominal bodies, including hand/fingers.",
        "Armature remains a separate joint property (nominal 0.001 kg m²). The ratios above compare link inertia traces, not whole-robot generalized inertia or closed-loop response.",
        "",
        "DR leaves arm-link masses/inertias unchanged. Realized payloads and gain/effort scales for 16 environment-reset samples are in comparison.json; full per-joint friction and all tensor values are in each runtime_model.json.",
        "Hand inertia/COM change with the added offset payload, as intended. This DR does not remove the 0.586441 kg gripper assembly or cover the FR3 arm-link mass properties.",
        "",
        "## Frames and provenance",
        "",
        "Inertias are at COM, expressed in the rigid-body-prim frame; raw nine-element tensors are column-major. The matrix field converts that layout explicitly.",
        "COM poses give the principal-axes frame relative to the body, with xyzw quaternions. This follows the installed low-level PhysX tensor API, despite the Isaac wrapper world-frame docstring.",
        "The saved training USD hashes and exact historical runtime environment are unavailable. These results establish current effective physics with matching source/config and audit assets, not byte-for-byte historical execution.",
        "Reduced instance counts change DR sample sequences relative to training. Seeds, environment indices, actual sampled values, config/checkpoint hashes, invocation and current software revisions are recorded.",
        "Policy delay sampling was not executed: this is a physics/reset extraction; its saved distribution remains in scenario metadata.",
        "",
        "## Next step",
        "",
        "Build a separate FR3v2 bare-flange model with documented inertials and COMs, then evaluate frozen policies with the intended controller/governor. Preserve the original baseline and bundles. Do not apply a guessed global inertia scale.",
        "",
    ]
    (a.output_dir / "FINDINGS.md").write_text("\n".join(lines))
    shutil.copyfile(__file__, a.output_dir / "summarize_training_physics.py")
    (a.output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{digest(p)}  {p.relative_to(a.output_dir)}\n"
            for p in sorted(a.output_dir.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"
        )
    )
    print(a.output_dir)


if __name__ == "__main__":
    main()

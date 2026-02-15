#!/usr/bin/env python3
"""Post-process MuJoCo MJCF to ensure quaternions, tweak defaults, and add sensors."""

from __future__ import annotations

import math
import os
import re
import sys
import xml.etree.ElementTree as ET

try:
    from transforms3d.euler import eul2quat as _eul2quat
except Exception:  # pragma: no cover - optional dependency
    _eul2quat = None


def _euler_to_quat(ex: float, ey: float, ez: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (x, y, z) in radians to quaternion (w, x, y, z).

    Uses static XYZ (extrinsic) rotation sequence, matching transforms3d 'sxyz'.
    """
    if _eul2quat is not None:
        w, x, y, z = _eul2quat(ex, ey, ez, axes="sxyz")
        return w, x, y, z

    # Fallback: construct quaternion from static XYZ rotations.
    cx = math.cos(ex * 0.5)
    sx = math.sin(ex * 0.5)
    cy = math.cos(ey * 0.5)
    sy = math.sin(ey * 0.5)
    cz = math.cos(ez * 0.5)
    sz = math.sin(ez * 0.5)

    # R = Rz * Ry * Rx
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx

    return w, x, y, z


def _fmt(val: float) -> str:
    if abs(val) < 1e-12:
        val = 0.0
    return f"{val:.15g}"


def _replace_euler_with_quat(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        parts = raw.split()
        if len(parts) != 3:
            return match.group(0)
        try:
            ex, ey, ez = (float(p) for p in parts)
        except ValueError:
            return match.group(0)
        w, x, y, z = _euler_to_quat(ex, ey, ez)
        return f'quat="{_fmt(w)} {_fmt(x)} {_fmt(y)} {_fmt(z)}"'

    return re.sub(r'euler="([^"]+)"', repl, text)


def _update_default_position(text: str, kp: str, dampratio: str) -> str:
    if "<worldbody>" not in text:
        return text

    pre, post = text.split("<worldbody>", 1)
    match = re.search(r"<position\b[^>]*>", pre)
    if not match:
        return text

    tag = match.group(0)

    def set_attr(tag_text: str, attr: str, value: str) -> str:
        if re.search(rf'{attr}="[^"]*"', tag_text):
            return re.sub(rf'{attr}="[^"]*"', f'{attr}="{value}"', tag_text)
        # Insert before closing
        if tag_text.endswith("/>"):
            return tag_text[:-2] + f' {attr}="{value}"/>'
        return tag_text[:-1] + f' {attr}="{value}">' 

    updated = set_attr(tag, "kp", kp)
    updated = set_attr(updated, "dampratio", dampratio)

    return pre[: match.start()] + updated + pre[match.end() :] + "<worldbody>" + post


def _extract_actuator_pairs(text: str) -> list[tuple[str, str]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    actuator_root = root.find("actuator")
    if actuator_root is None:
        return []

    pairs: list[tuple[str, str]] = []
    for actuator in actuator_root:
        name = actuator.attrib.get("name")
        joint = actuator.attrib.get("joint")
        if name and joint:
            pairs.append((name, joint))
    return pairs


PASSIVE_JOINTS = {
    "trochanter0",
    "trochanter1",
    "patella0",
    "patella1",
}


def _remove_passive_actuators(text: str) -> str:
    match = re.search(r"<actuator>(.*?)</actuator>", text, re.DOTALL)
    if not match:
        return text

    block = match.group(1)
    lines = block.splitlines()
    filtered: list[str] = []
    for line in lines:
        joint_match = re.search(r'\bjoint="([^"]+)"', line)
        if joint_match and joint_match.group(1) in PASSIVE_JOINTS:
            continue
        filtered.append(line)

    new_block = "\n".join(filtered)
    return text[: match.start(1)] + new_block + text[match.end(1) :]


def _prune_actuator_sensors(text: str, actuator_names: set[str]) -> str:
    lines = text.splitlines()
    filtered: list[str] = []
    for line in lines:
        if "<actuatorfrc" in line:
            match = re.search(r'\bactuator="([^"]+)"', line)
            if match and match.group(1) not in actuator_names:
                continue
        filtered.append(line)
    return "\n".join(filtered) + ("\n" if text.endswith("\n") else "")


def _add_dual_actuators(text: str) -> str:
    match = re.search(r"<actuator>(.*?)</actuator>", text, re.DOTALL)
    if not match:
        return text

    block = match.group(1)
    tag_pattern = re.compile(r"<(position|motor)\b([^/>]*)/>")

    def parse_attrs(attr_text: str) -> dict[str, str]:
        return {k: v for k, v in re.findall(r'(\w+)="([^"]*)"', attr_text)}

    def format_tag(tag: str, attrs: dict[str, str]) -> str:
        order = ["class", "name", "joint", "forcerange", "ctrlrange", "inheritrange"]
        parts = []
        for key in order:
            if key in attrs:
                parts.append(f'{key}="{attrs[key]}"')
        for key in sorted(k for k in attrs.keys() if k not in order):
            parts.append(f'{key}="{attrs[key]}"')
        return f"    <{tag} " + " ".join(parts) + "/>"

    joints: dict[str, dict[str, dict[str, str]]] = {}
    names: set[str] = set()
    for tag, attr_text in tag_pattern.findall(block):
        attrs = parse_attrs(attr_text)
        name = attrs.get("name")
        joint = attrs.get("joint")
        if not name or not joint:
            continue
        names.add(name)
        joints.setdefault(joint, {})[tag] = attrs

    def unique_name(base: str) -> str:
        if base not in names:
            names.add(base)
            return base
        i = 2
        while f"{base}_{i}" in names:
            i += 1
        new_name = f"{base}_{i}"
        names.add(new_name)
        return new_name

    new_lines: list[str] = []
    for joint, kinds in joints.items():
        if "position" in kinds and "motor" not in kinds:
            pos_attrs = kinds["position"]
            motor_attrs: dict[str, str] = {
                "name": unique_name(f"{joint}_motor"),
                "joint": joint,
            }
            if "class" in pos_attrs:
                motor_attrs["class"] = pos_attrs["class"]
            if "forcerange" in pos_attrs:
                motor_attrs["forcerange"] = pos_attrs["forcerange"]
                motor_attrs["ctrlrange"] = pos_attrs["forcerange"]
            new_lines.append(format_tag("motor", motor_attrs))
        if "motor" in kinds and "position" not in kinds:
            motor_attrs = kinds["motor"]
            pos_attrs = {
                "name": unique_name(f"{joint}_pos"),
                "joint": joint,
                "inheritrange": "1",
            }
            if "class" in motor_attrs:
                pos_attrs["class"] = motor_attrs["class"]
            else:
                pos_attrs["class"] = "robot"
            if "forcerange" in motor_attrs:
                pos_attrs["forcerange"] = motor_attrs["forcerange"]
            new_lines.append(format_tag("position", pos_attrs))

    if not new_lines:
        return text

    insertion = block.rstrip() + "\n" + "\n".join(new_lines) + "\n"
    return text[: match.start(1)] + insertion + text[match.end(1) :]


def _ensure_sensor_block(text: str, actuator_pairs: list[tuple[str, str]]) -> str:
    required_lines = [
        "    <gyro name=\"imu_gyro\" site=\"imu\"/>",
        "    <accelerometer name=\"imu_accel\" site=\"imu\"/>",
    ]

    for actuator_name, joint_name in actuator_pairs:
        required_lines.append(
            f"    <jointvel name=\"jointvel_{joint_name}\" joint=\"{joint_name}\"/>"
        )
    for actuator_name, _joint_name in actuator_pairs:
        required_lines.append(
            f"    <actuatorfrc name=\"actuatorfrc_{actuator_name}\" actuator=\"{actuator_name}\"/>"
        )

    def sensor_name(line: str) -> str | None:
        match = re.search(r'name="([^"]+)"', line)
        return match.group(1) if match else None

    missing_lines = []
    for line in required_lines:
        name = sensor_name(line)
        if name is None:
            continue
        if f'name="{name}"' not in text:
            missing_lines.append(line)

    if not missing_lines:
        return text

    if "<sensor>" in text:
        insertion = "\n".join(missing_lines) + "\n  </sensor>"
        return re.sub(r"</sensor>", insertion, text, count=1)

    sensor_block = "\n".join(["  <sensor>"] + missing_lines + ["  </sensor>", ""])

    if "</actuator>" in text:
        return text.replace("</actuator>\n", "</actuator>\n" + sensor_block, 1)

    if "</mujoco>" in text:
        return text.replace("</mujoco>", sensor_block + "</mujoco>")

    return text


def main() -> int:
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        xml_path = os.path.join(os.path.dirname(__file__), "robot.xml")

    if not os.path.exists(xml_path):
        print(f"ERROR: MJCF file not found: {xml_path}", file=sys.stderr)
        return 1

    with open(xml_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = _replace_euler_with_quat(text)
    text = _update_default_position(text, kp="1", dampratio="0.9")
    text = _remove_passive_actuators(text)
    text = _add_dual_actuators(text)
    actuator_pairs = _extract_actuator_pairs(text)
    text = _ensure_sensor_block(text, actuator_pairs)
    actuator_names = {name for name, _joint in actuator_pairs}
    text = _prune_actuator_sensors(text, actuator_names)

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

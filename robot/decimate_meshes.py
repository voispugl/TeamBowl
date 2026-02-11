#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import trimesh

Backend = Literal["auto", "quadric", "cluster"]

SPECIAL_TARGET_RATIOS = {
    "wheel.stl": 0.05,
    "lid_gear.stl": 0.2,
}

SPECIAL_MIN_FACES = {
    "lid_gear.stl": 200,
}

WHEEL_COLLISION_SECTIONS = 16


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Scene is empty: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type for {path}: {type(loaded)}")

    mesh = loaded.copy()
    mesh.remove_infinite_values()
    mesh.merge_vertices()
    return mesh


def _cluster_decimate_once(mesh: trimesh.Trimesh, cell_size: float) -> trimesh.Trimesh | None:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if len(verts) == 0 or len(faces) == 0:
        return None

    min_v = verts.min(axis=0)
    grid = np.floor((verts - min_v) / max(cell_size, 1e-12)).astype(np.int64)

    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    if n_clusters < 4:
        return None

    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    new_vertices = np.zeros((n_clusters, 3), dtype=np.float64)
    for axis in range(3):
        sums = np.bincount(inverse, weights=verts[:, axis], minlength=n_clusters)
        new_vertices[:, axis] = sums / np.maximum(counts, 1.0)

    remapped = inverse[faces]

    non_deg = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[non_deg]
    if len(remapped) == 0:
        return None

    canonical = np.sort(remapped, axis=1)
    _, unique_idx = np.unique(canonical, axis=0, return_index=True)
    remapped = remapped[np.sort(unique_idx)]

    dec = trimesh.Trimesh(vertices=new_vertices, faces=remapped, process=False)
    dec.remove_infinite_values()
    dec.remove_unreferenced_vertices()
    dec.update_faces(dec.nondegenerate_faces())
    dec.update_faces(dec.unique_faces())
    dec.remove_unreferenced_vertices()
    dec.merge_vertices()

    if len(dec.faces) == 0:
        return None
    return dec


def _cluster_decimate(mesh: trimesh.Trimesh, target_faces: int, max_iters: int = 12) -> trimesh.Trimesh:
    orig_faces = len(mesh.faces)
    if orig_faces <= target_faces:
        return mesh.copy()

    verts = np.asarray(mesh.vertices)
    bbox = verts.max(axis=0) - verts.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    if diag <= 1e-12:
        return mesh.copy()

    low = diag / 4096.0
    high = diag / 2.0

    best_any: trimesh.Trimesh | None = None
    best_any_diff = float("inf")

    best_above: trimesh.Trimesh | None = None
    best_above_faces = 10**18

    best_below: trimesh.Trimesh | None = None
    best_below_faces = -1

    for _ in range(max_iters):
        cell_size = float(np.sqrt(low * high))
        candidate = _cluster_decimate_once(mesh, cell_size)
        if candidate is None:
            high = max(low * 1.1, high * 0.8)
            continue

        candidate_faces = int(len(candidate.faces))
        diff = abs(candidate_faces - target_faces)
        if diff < best_any_diff:
            best_any = candidate
            best_any_diff = diff

        if candidate_faces >= target_faces:
            if candidate_faces < best_above_faces:
                best_above = candidate
                best_above_faces = candidate_faces
            low = cell_size
        else:
            if candidate_faces > best_below_faces:
                best_below = candidate
                best_below_faces = candidate_faces
            high = cell_size

    if best_above is not None and best_below is not None:
        above_diff = abs(best_above_faces - target_faces)
        below_diff = abs(best_below_faces - target_faces)
        return best_above if above_diff <= below_diff else best_below
    if best_above is not None:
        return best_above
    if best_below is not None:
        return best_below
    return best_any if best_any is not None else mesh.copy()


def _quadric_decimate(mesh: trimesh.Trimesh, target_faces: int, aggression: int = 7) -> trimesh.Trimesh:
    return mesh.simplify_quadric_decimation(face_count=target_faces, aggression=aggression)


def _decimate(
    mesh: trimesh.Trimesh,
    target_faces: int,
    backend: Backend,
    quadric_aggression: int,
) -> tuple[trimesh.Trimesh, str]:
    if backend in ("auto", "quadric"):
        try:
            dec = _quadric_decimate(mesh, target_faces=target_faces, aggression=quadric_aggression)
            if len(dec.faces) > 0:
                return dec, "quadric"
        except Exception as exc:
            if backend == "quadric":
                raise RuntimeError(
                    "Quadric decimation failed. Install `fast-simplification` or use --backend cluster"
                ) from exc

    dec = _cluster_decimate(mesh, target_faces=target_faces)
    return dec, "cluster"


def _make_wheel_collision_mesh(mesh: trimesh.Trimesh, sections: int) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        raise ValueError("Wheel mesh has no vertices.")

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) * 0.5
    extents = maxs - mins

    radius = 0.5 * float(max(extents[1], extents[2]))
    height = float(extents[0])

    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0])
    cyl.apply_transform(rot)
    cyl.apply_translation(center)
    cyl.remove_infinite_values()
    cyl.merge_vertices()
    return cyl


def _iter_meshes(assets_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(p for p in assets_dir.glob(pattern) if p.is_file())


def _rewrite_xml_mesh_paths(xml_path: Path, new_mesh_subdir: str) -> None:
    text = xml_path.read_text(encoding="utf-8")

    mesh_pattern = r'(<mesh[^>]*\sfile\s*=\s*")([^"]+)(")'

    def repl(m: re.Match[str]) -> str:
        src = m.group(2)
        base = Path(src).name
        return f'{m.group(1)}{new_mesh_subdir}/{base}{m.group(3)}'

    replaced, count = re.subn(mesh_pattern, repl, text)
    if count == 0:
        raise ValueError(f"No <mesh file=...> entries found in {xml_path}")

    # Ensure compiler meshdir doesn't conflict with explicit mesh file paths.
    replaced = re.sub(r'\smeshdir\s*=\s*"[^"]*"', "", replaced)

    xml_path.write_text(replaced, encoding="utf-8")


def _set_xml_meshdir(xml_path: Path, new_meshdir: str) -> None:
    text = xml_path.read_text(encoding="utf-8")
    pattern = r'(<compiler[^>]*\smeshdir\s*=\s*")[^"]+(")'
    replaced, count = re.subn(
        pattern,
        lambda m: f"{m.group(1)}{new_meshdir}{m.group(2)}",
        text,
        count=1,
    )
    if count == 0:
        raise ValueError(f"No compiler meshdir attribute found in {xml_path}")
    xml_path.write_text(replaced, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch decimate robot meshes.")
    parser.add_argument("--assets_dir", default=str(Path(__file__).resolve().parent / "assets"))
    parser.add_argument("--pattern", default="*.stl", help="Glob within assets_dir, e.g. '*.stl'")
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.35,
        help="Target face ratio (0-1], e.g. 0.35 keeps ~35%% of faces.",
    )
    parser.add_argument(
        "--min_faces",
        type=int,
        default=300,
        help="Never decimate below this many faces per mesh.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "quadric", "cluster"),
        default="auto",
        help="Decimation backend. 'auto' tries quadric then falls back to cluster.",
    )
    parser.add_argument(
        "--quadric_aggression",
        type=int,
        default=7,
        help="Aggression for quadric decimation when available.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "assets_decimated"),
        help="Directory for decimated meshes when not using --inplace.",
    )
    parser.add_argument("--inplace", action="store_true", help="Overwrite meshes in assets_dir.")
    parser.add_argument("--dry_run", action="store_true", help="Compute stats but do not write files.")
    parser.add_argument(
        "--backup_dir",
        default="",
        help="Optional backup dir used only with --inplace.",
    )
    parser.add_argument(
        "--rewrite_xml",
        default="",
        help=(
            "Optional path to robot.xml. If set and not inplace, each <mesh file=...> "
            "is rewritten to point at output_dir."
        ),
    )
    parser.add_argument(
        "--rewrite_mode",
        choices=("mesh_paths", "meshdir"),
        default="mesh_paths",
        help="How --rewrite_xml updates XML references.",
    )

    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).expanduser().resolve()
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets dir not found: {assets_dir}")

    output_dir = assets_dir if args.inplace else Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.inplace and args.backup_dir:
        backup_dir = Path(args.backup_dir).expanduser().resolve()
        if not args.dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)
    else:
        backup_dir = None

    mesh_paths = list(_iter_meshes(assets_dir, args.pattern))
    if not mesh_paths:
        raise RuntimeError(f"No meshes matched {args.pattern} in {assets_dir}")

    total_before = 0
    total_after = 0

    print(f"Found {len(mesh_paths)} meshes in {assets_dir}")
    print(f"Decimation backend: {args.backend}")

    for src in mesh_paths:
        mesh = _load_mesh(src)
        faces_before = int(len(mesh.faces))
        target_ratio = SPECIAL_TARGET_RATIOS.get(src.name, args.target_ratio)
        min_faces = SPECIAL_MIN_FACES.get(src.name, args.min_faces)
        target_faces = max(min_faces, int(round(faces_before * target_ratio)))

        decimated, used_backend = _decimate(
            mesh,
            target_faces=target_faces,
            backend=args.backend,
            quadric_aggression=args.quadric_aggression,
        )

        faces_after = int(len(decimated.faces))
        if faces_after >= faces_before:
            # If simplification failed to reduce, keep original mesh.
            decimated = mesh
            faces_after = faces_before
            used_backend = f"{used_backend} (no_change)"

        total_before += faces_before
        total_after += faces_after

        dst = output_dir / src.name

        if not args.dry_run:
            if args.inplace and backup_dir is not None:
                shutil.copy2(src, backup_dir / src.name)
            decimated.export(dst)

        pct = 100.0 * (1.0 - (faces_after / max(faces_before, 1)))
        ratio_note = ""
        if target_ratio != args.target_ratio or min_faces != args.min_faces:
            ratio_note = f" (target_ratio={target_ratio:.2f}, min_faces={min_faces})"
        print(
            f"{src.name}: {faces_before} -> {faces_after} faces "
            f"({pct:.1f}% reduction) [{used_backend}]{ratio_note}"
        )

        if src.name == "wheel.stl":
            wheel_collision = _make_wheel_collision_mesh(mesh, WHEEL_COLLISION_SECTIONS)
            if not args.dry_run:
                wheel_collision.export(output_dir / "wheel_collision.stl")
            print(
                f"wheel_collision.stl: {len(wheel_collision.faces)} faces "
                f"(sections={WHEEL_COLLISION_SECTIONS})"
            )

    total_pct = 100.0 * (1.0 - (total_after / max(total_before, 1)))
    print(f"Total: {total_before} -> {total_after} faces ({total_pct:.1f}% reduction)")

    if args.rewrite_xml:
        xml_path = Path(args.rewrite_xml).expanduser().resolve()
        if args.inplace:
            print("--rewrite_xml ignored with --inplace.")
        else:
            rel_meshdir = os.path.relpath(output_dir, xml_path.parent)
            if args.dry_run:
                if args.rewrite_mode == "mesh_paths":
                    print(f"Would rewrite mesh file paths in {xml_path} to '{rel_meshdir}/*.stl'")
                else:
                    print(f"Would update compiler meshdir in {xml_path} to '{rel_meshdir}'")
            else:
                if args.rewrite_mode == "mesh_paths":
                    _rewrite_xml_mesh_paths(xml_path, rel_meshdir)
                    print(f"Rewrote mesh file paths in {xml_path} to '{rel_meshdir}/*.stl'")
                else:
                    _set_xml_meshdir(xml_path, rel_meshdir)
                    print(f"Updated compiler meshdir in {xml_path} to '{rel_meshdir}'")


if __name__ == "__main__":
    main()

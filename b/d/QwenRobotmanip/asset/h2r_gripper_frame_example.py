#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the paper's Human-to-Robot Action Alignment geometry
(data.tex L143-159): how a parallel-jaw gripper action a_t = (p, R, w) is
constructed from MANO 3D hand keypoints.

Left panel  : a 3D scene with the MANO keypoints (wrist, thumb tip, index tip,
              middle tip), the virtual finger k_vf = 0.7*index + 0.3*middle,
              the end-effector position p = midpoint(thumb, k_vf), the gripper
              width w = ||thumb - k_vf||, and the constructed right-handed
              orthonormal frame R = [x y z] anchored at p, where
              z = grasp axis (jaw line), y = gripper normal, x = approach.
Right panel : a schematic parallel-jaw gripper showing the same axes and the
              opening w, plus the construction formulas.

Run:
    /mnt/r/VENV/dj/bin/python h2r_gripper_frame_example.py
Output:
    h2r_gripper_frame_example.png (same directory, same base name)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from mpl_toolkits.mplot3d.proj3d import proj_transform  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

COLOR_X = "#1f77b4"  # blue : approach axis x
COLOR_Y = "#ff7f0e"  # orange: gripper-normal axis y
COLOR_Z = "#2ca02c"  # green : grasp axis z (jaw line)
COLOR_KP = "#444444"  # keypoints
COLOR_VF = "#9467bd"  # virtual finger
COLOR_P = "#d62728"  # EEF position p


class Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch that can be projected/drawn in 3D axes."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs2d, ys2d, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return min(zs3d)


def add_vector(ax, origin, vec, color, lw=2.8, ls="-", mutation_scale=16):
    arrow = Arrow3D(
        [origin[0], origin[0] + vec[0]],
        [origin[1], origin[1] + vec[1]],
        [origin[2], origin[2] + vec[2]],
        mutation_scale=mutation_scale,
        lw=lw,
        arrowstyle="-|>",
        color=color,
        linestyle=ls,
    )
    ax.add_artist(arrow)


def retarget(k_thumb, k_index, k_middle, k_wrist, s=+1.0):
    """Paper eqs (data.tex L146-158)."""
    eps = 1e-9
    k_vf = 0.7 * k_index + 0.3 * k_middle
    p = 0.5 * (k_thumb + k_vf)
    w = np.linalg.norm(k_thumb - k_vf)
    z = s * (k_thumb - k_vf) / max(w, eps)
    d = k_vf - k_wrist
    y = np.cross(z, d)
    y = y / max(np.linalg.norm(y), eps)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)  # columns = x, y, z
    return k_vf, p, w, x, y, z, d, R


def draw_left_panel(ax, kp, res):
    k_thumb, k_index, k_middle, k_wrist = kp
    k_vf, p, w, x, y, z, d, R = res

    # scale of the drawn frame axes (relative to gripper width)
    a = max(w, 0.02) * 0.9

    # keypoints (short tags, offset to reduce overlap)
    kp_items = [
        ("wrist", k_wrist, (-0.010, -0.006, -0.010)),
        ("thumb", k_thumb, (0.004, 0.000, 0.012)),
        ("index", k_index, (0.004, 0.006, 0.006)),
        ("middle", k_middle, (0.004, 0.008, -0.012)),
    ]
    for name, pt, off in kp_items:
        ax.scatter(*pt, color=COLOR_KP, s=42, depthshade=False)
        ax.text(pt[0] + off[0], pt[1] + off[1], pt[2] + off[2], name, color=COLOR_KP, fontsize=9)

    # virtual finger and p (short tags)
    ax.scatter(*k_vf, color=COLOR_VF, s=110, marker="*", depthshade=False)
    ax.text(k_vf[0] - 0.004, k_vf[1] + 0.004, k_vf[2] - 0.016, r"$k_{vf}$", color=COLOR_VF, fontsize=11, weight="bold")
    ax.scatter(*p, color=COLOR_P, s=75, depthshade=False)
    ax.text(p[0] + 0.006, p[1] - 0.004, p[2] + 0.006, r"$p$", color=COLOR_P, fontsize=12, weight="bold")

    # jaw line (thumb - vf), i.e. width w
    ax.plot([k_thumb[0], k_vf[0]], [k_thumb[1], k_vf[1]], [k_thumb[2], k_vf[2]],
            color=COLOR_Z, lw=1.8, ls="--")
    mid_tv = 0.5 * (k_thumb + k_vf)
    ax.text(mid_tv[0], mid_tv[1] - 0.010, mid_tv[2] + 0.010, rf"$w={w:.3f}$ m", color=COLOR_Z, fontsize=9)

    # wrist -> vf direction d (dashed gray)
    ax.plot([k_wrist[0], k_vf[0]], [k_wrist[1], k_vf[1]], [k_wrist[2], k_vf[2]],
            color="#999999", lw=1.1, ls=":")
    mid_wv = 0.5 * (k_wrist + k_vf)
    ax.text(mid_wv[0] - 0.006, mid_wv[1] + 0.004, mid_wv[2] + 0.008, r"$d$", color="#777777", fontsize=10)

    # gripper frame at p
    add_vector(ax, p, x * a, COLOR_X)
    add_vector(ax, p, y * a, COLOR_Y)
    add_vector(ax, p, z * a, COLOR_Z)
    ax.text(*(p + x * a * 1.12), r"$x$ approach", color=COLOR_X, fontsize=10, weight="bold")
    ax.text(*(p + y * a * 1.12), r"$y$ normal", color=COLOR_Y, fontsize=10, weight="bold")
    ax.text(*(p + z * a * 1.12), r"$z$ grasp", color=COLOR_Z, fontsize=10, weight="bold")

    # limits
    pts = np.stack([k_thumb, k_index, k_middle, k_wrist, k_vf, p], axis=0)
    lo = pts.min(axis=0) - a * 1.3
    hi = pts.max(axis=0) + a * 1.3
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=20, azim=-72)
    ax.set_title(
        "Action Alignment: build gripper frame R=[x y z] from MANO keypoints\n"
        "z=grasp axis (jaw line), y=gripper normal, x=approach (right-handed)",
        fontsize=11,
        pad=12,
    )


def draw_right_panel(ax, res):
    k_vf, p, w, x, y, z, d, R = res
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Schematic parallel-jaw gripper (2D): grasp axis z vertical, approach x horizontal.
    # p at center; two jaws symmetric about p along z, separated by opening w; palm behind (-x).
    cx, cy = 4.6, 6.2
    half = 1.5           # half opening (drawn)
    jaw_len = 1.7        # finger length along +x (approach)
    jaw_th = 0.35        # finger thickness
    # palm / body (behind p, spanning the opening)
    ax.add_patch(Rectangle((cx - 0.5, cy - half - jaw_th), 0.5, 2 * (half + jaw_th),
                           facecolor="#cccccc", edgecolor="black"))
    # two jaws (fingers) pointing +x, symmetric about cy
    ax.add_patch(Rectangle((cx, cy + half), jaw_len, jaw_th, facecolor="#b0d0ec", edgecolor=COLOR_X))
    ax.add_patch(Rectangle((cx, cy - half - jaw_th), jaw_len, jaw_th, facecolor="#b0d0ec", edgecolor=COLOR_X))
    # opening arrow (width w) along z, between inner jaw faces
    ax.annotate("", xy=(cx + jaw_len + 0.3, cy + half), xytext=(cx + jaw_len + 0.3, cy - half),
                arrowprops=dict(arrowstyle="<->", color=COLOR_Z, lw=2))
    ax.text(cx + jaw_len + 0.45, cy - 0.1, rf"$w={w:.3f}$ m" "\n(opening)", color=COLOR_Z, fontsize=10, va="center")
    # axes at p
    ax.annotate("", xy=(cx, cy + 2.2), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color=COLOR_Z, lw=2.4))
    ax.text(cx + 0.12, cy + 2.25, r"$z$ grasp", color=COLOR_Z, fontsize=11, weight="bold")
    ax.annotate("", xy=(cx + 2.4, cy), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color=COLOR_X, lw=2.4))
    ax.text(cx + 1.4, cy - 0.45, r"$x$ approach", color=COLOR_X, fontsize=11, weight="bold")
    ax.scatter([cx], [cy], color=COLOR_P, s=70, zorder=5)
    ax.text(cx - 1.35, cy + 0.05, r"$p$", color=COLOR_P, fontsize=12, weight="bold")

    # formula box
    formula = (
        r"$k_{vf}=0.7\,k_{index}+0.3\,k_{middle}$" "\n"
        r"$p=\frac{1}{2}(k_{thumb}+k_{vf})$,   $w=\|k_{thumb}-k_{vf}\|$" "\n"
        r"$z=\dfrac{s\,(k_{thumb}-k_{vf})}{w}$,   $d=k_{vf}-k_{wrist}$" "\n"
        r"$y=\dfrac{z\times d}{\|z\times d\|}$,   $x=y\times z$" "\n"
        r"$s=+1$ (right hand),  $s=-1$ (left hand)"
    )
    ax.text(0.2, 2.3, formula, fontsize=10.5, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#999999"))
    ax.set_title("Parallel-jaw gripper: p, R=[x y z], width w", fontsize=11, pad=12)


def main():
    # A representative right-hand frame (metric, meters). Numbers chosen so the
    # geometry is clearly 3D and the constructed frame is non-degenerate.
    k_wrist = np.array([0.00, 0.00, 0.00])
    k_thumb = np.array([0.085, 0.020, 0.030])
    k_index = np.array([0.070, 0.075, 0.000])
    k_middle = np.array([0.055, 0.080, -0.015])
    kp = (k_thumb, k_index, k_middle, k_wrist)
    res = retarget(*kp, s=+1.0)

    k_vf, p, w, x, y, z, d, R = res
    print("k_vf =", np.round(k_vf, 4))
    print("p    =", np.round(p, 4), " w =", round(float(w), 4))
    print("x    =", np.round(x, 4))
    print("y    =", np.round(y, 4))
    print("z    =", np.round(z, 4))
    print("det(R) =", round(float(np.linalg.det(R)), 6), " (should be +1)")

    fig = plt.figure(figsize=(15, 7.0))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2)
    draw_left_panel(ax_left, kp, res)
    draw_right_panel(ax_right, res)

    fig.suptitle(
        "Human-to-Robot Action Alignment: MANO hand keypoints -> parallel-jaw gripper action (p, R, w)",
        fontsize=13.5,
        weight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h2r_gripper_frame_example.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

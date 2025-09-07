from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def standing_success(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.9,
    upright_threshold: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Success when base height and uprightness exceed thresholds."""
    robot: RigidObject = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    # uprightness proxy from projected gravity (small x,y magnitude => upright)
    g_b = robot.data.projected_gravity_b
    tilt_mag = torch.linalg.norm(g_b[:, :2], dim=1)
    upright = torch.exp(-0.5 * (tilt_mag / 0.3) ** 2)
    return (z > height_threshold) & (upright > upright_threshold)


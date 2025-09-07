from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def uprightness_exp(env: ManagerBasedRLEnv, std: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward uprightness using projected gravity on the base frame.

    Computes alignment of base z-axis to world z via projected gravity magnitude on x/y.
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    # projected gravity in base frame (x,y,z) where desired is [0, 0, -g]
    g_b = robot.data.projected_gravity_b  # (N, 3)
    # magnitude of x,y components indicates tilt; smaller is better
    tilt_mag = torch.linalg.norm(g_b[:, :2], dim=1)
    return torch.exp(-0.5 * (tilt_mag / std) ** 2)


def base_height_tanh(env: ManagerBasedRLEnv, target: float, std: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Encourage base height toward a target with tanh kernel."""
    robot: RigidObject = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    return 1.0 - torch.tanh(torch.abs(z - target) / std)


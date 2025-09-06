 from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    distance_threshold: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward when the cube is above minimal height and within reach of the end-effector.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # World positions
    cube_pos = cube.data.root_pos_w  # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # use gripper target (index 0)

    height_ok = cube_pos[:, 2] > minimal_height
    dist = torch.norm(cube_pos - ee_pos, dim=1)
    reach_ok = dist < distance_threshold

    return (height_ok & reach_ok).float()


def cube_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.03,
    gripper_close_threshold: float = 0.6,
) -> torch.Tensor:
    """
    Reward if the cube is grasped: close to EE and gripper closed.

    Notes:
    - Uses jaw frame (index 1) if available for proximity check to be conservative.
    - For gripper closure, tries last two joints; if only one joint exists, uses that.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    # Distance between EE and cube
    cube_pos = cube.data.root_pos_w
    # prefer jaw target (index 1) if present else fall back to index 0
    ee_targets = ee_frame.data.target_pos_w
    if ee_targets.shape[-2] > 1:
        ee_pos = ee_targets[:, 1, :]
    else:
        ee_pos = ee_targets[:, 0, :]
    pose_diff = torch.linalg.vector_norm(cube_pos - ee_pos, dim=1)

    # Gripper closure heuristic
    joint_pos = robot.data.joint_pos
    if joint_pos.shape[1] >= 2:
        gripper_closed = torch.logical_and(
            joint_pos[:, -1] >= gripper_close_threshold,
            joint_pos[:, -2] >= gripper_close_threshold,
        )
    else:
        gripper_closed = joint_pos[:, -1] >= gripper_close_threshold

    return torch.logical_and(pose_diff < diff_threshold, gripper_closed).float()


def cube_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Gaussian-shaped reward on EE-to-cube distance."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return torch.exp(-0.5 * (distance / std) ** 2)


def cube_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using a tanh-kernel once lifted."""
    robot: RigidObject = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - cube.data.root_pos_w, dim=1)

    lifted = cube.data.root_pos_w[:, 2] > minimal_height
    return lifted * (1 - torch.tanh(distance / std))


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sanitize_reward(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    distance_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the agent for lifting the object above a minimal height.

    *Only if* it is within a certain distance from the end-effector.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Get object height (z position in world frame)
    obj_height = object.data.root_pos_w[:, 2]  # (num_envs,)

    # Get positions
    obj_pos = object.data.root_pos_w  # (num_envs, 3)
    # Use average if multiple EE target frames are provided (e.g., fingertips)
    ee_targets = ee_frame.data.target_pos_w.permute(1,0,2)

    # Compute Euclidean distance between object and end-effector
    dist = torch.norm(obj_pos - ee_targets, dim=-1)[0]  # (num_envs,)
    # print("Dist", dist)

    # Reward is 1.0 if object is above minimal height AND within_reach to EE
    lifted = obj_height > minimal_height
    within_reach = dist < distance_threshold

    reward = torch.where(lifted & within_reach, 1.0, 0.0)

    return reward


def gripper_open_approach(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    open_threshold: float = 0.4,
    sharpness: float = 25.0,
) -> torch.Tensor:
    """Reward keeping gripper open when far from the object.

    - High when EE-object distance > distance_threshold AND gripper joint indicates open (> open_threshold).
    - Smoothly decays near the object to avoid "stuck open" conflicts with closing rewards.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # distance gate
    ee_targets = ee_frame.data.target_pos_w
    ee_pos = ee_targets.mean(dim=-2) if ee_targets.shape[-2] > 1 else ee_targets[:, 0, :]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w - ee_pos, dim=1)
    far_gate = torch.sigmoid((dist - distance_threshold) * sharpness)

    # gripper open gate (SO-ARM100: higher joint value is more open)
    grip = robot.data.joint_pos[:, -1]
    open_gate = torch.sigmoid((grip - open_threshold) * sharpness)

    return _sanitize_reward(far_gate * open_gate)


def gripper_close_when_near(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.06,
    close_threshold: float = 0.2,
    closed_below: bool = True,
    sharpness: float = 25.0,
) -> torch.Tensor:
    """Reward closing the gripper when near the object (smooth, distance-gated).

    - High when EE-object distance < distance_threshold and gripper indicates closed toward the object.
    - ``closed_below`` selects whether lower joint values indicate more closed (e.g., SO-ARM100).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # distance gate (near -> high)
    ee_targets = ee_frame.data.target_pos_w
    ee_pos = ee_targets.mean(dim=-2) if ee_targets.shape[-2] > 1 else ee_targets[:, 0, :]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w - ee_pos, dim=1)
    near_gate = torch.sigmoid((distance_threshold - dist) * sharpness)

    # gripper closure gate
    grip = robot.data.joint_pos[:, -1]
    if closed_below:
        close_gate = torch.sigmoid((close_threshold - grip) * sharpness)
    else:
        close_gate = torch.sigmoid((grip - close_threshold) * sharpness)

    return _sanitize_reward(near_gate * close_gate)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.3,  # standard deviation
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_targets = object.data.root_pos_w
    cube_pos_w = ee_frame.data.target_pos_w.permute(1,0,2)

    distance = torch.linalg.vector_norm(cube_pos_w - object_targets, dim=-1)[0]
    # print("Distance", distance)

    reward = torch.exp(-0.5 * (distance / std) ** 2)

    return reward


def object_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.03,
    gripper_close_threshold: float = 0.2,
) -> torch.Tensor:
    """
    Reward function for detecting if the object is being grasped.

    Combines end-effector proximity and gripper closure conditions.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Compute the distance between end-effector and object
    object_targets = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w.permute(1,0,2)

    pose_diff = torch.linalg.vector_norm(object_targets - end_effector_pos, dim=-1)[0]
    # print("Pose_diff", pose_diff)
    # Check if gripper joints are closed beyond threshold
    gripper_closed = robot.data.joint_pos[:, -1] <= gripper_close_threshold

    # Combine both conditions
    is_grasped = torch.logical_and(pose_diff < diff_threshold, gripper_closed)

    # print(f"object grasp reward: {is_grasped.float()}")

    return is_grasped.float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_goal_distance_ground_level(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for moving object toward goal even on the ground.
    
    This provides reward for goal tracking before the object is lifted,
    helping bootstrap the learning process.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the object to goal (only X-Y, ignore Z)
    object_pos_xy = object.data.root_pos_w[:, :2]
    goal_pos_xy = des_pos_w[:, :2]
    distance_xy = torch.norm(goal_pos_xy - object_pos_xy, dim=1)
    # reward decreases with distance (no height requirement)
    return 1 - torch.tanh(distance_xy / std)

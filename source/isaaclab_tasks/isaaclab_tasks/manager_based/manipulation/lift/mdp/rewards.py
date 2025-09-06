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


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float | None = None,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height.

    If ``distance_threshold`` is provided, also requires the object to be within that
    distance of the end-effector (to avoid rewarding accidental bounces).
    """
    object: RigidObject = env.scene[object_cfg.name]
    height_ok = object.data.root_pos_w[:, 2] > minimal_height

    if distance_threshold is None:
        return height_ok.float()

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos = object.data.root_pos_w
    ee_targets = ee_frame.data.target_pos_w
    if ee_targets.shape[-2] > 1:
        ee_pos = 0.5 * (ee_targets[:, 0, :] + ee_targets[:, 1, :])
    else:
        ee_pos = ee_targets[:, 0, :]
    within_reach = torch.norm(obj_pos - ee_pos, dim=1) < distance_threshold
    return (height_ok & within_reach).float()


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_targets = ee_frame.data.target_pos_w
    if ee_targets.shape[-2] > 1:
        ee_w = 0.5 * (ee_targets[:, 0, :] + ee_targets[:, 1, :])
    else:
        ee_w = ee_targets[:, 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.02,
    gripper_joint_name: str = "Gripper",
    gripper_close_command: float = 0.5,
    gripper_open_command: float = 0.0,
    close_margin: float = 0.02,
    gripper_term_name: str = "gripper_action",
) -> torch.Tensor:
    """Reward term for grasping the object.

    - Proximity: object close to EE (within ``diff_threshold``)
    - Closure: gripper joint indicates closed state (less-than if ``closed_below`` else greater-than)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos = obj.data.root_pos_w
    ee_targets = ee_frame.data.target_pos_w
    if ee_targets.shape[-2] > 1:
        ee_pos = 0.5 * (ee_targets[:, 0, :] + ee_targets[:, 1, :])
    else:
        ee_pos = ee_targets[:, 0, :]
    pose_diff = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)

    # find gripper joint index by name; fallback to last joint if not found
    try:
        idx = robot.data.joint_names.index(gripper_joint_name)
    except ValueError:
        idx = -1
    grip_val = robot.data.joint_pos[:, idx]

    # Check the last gripper action (negative => close for BinaryJointAction)
    try:
        gripper_term = env.action_manager.get_term(gripper_term_name)
        raw = gripper_term.raw_actions.squeeze(-1)
        is_closing = raw < 0
    except Exception:
        # Fallback: assume closing when near object if term not found
        is_closing = torch.ones_like(grip_val, dtype=torch.bool)

    within_reach = pose_diff < diff_threshold
    # Object between tips if joint can't reach full close but is not open
    partially_closed = (grip_val > (gripper_open_command + close_margin)) & (grip_val < (gripper_close_command - close_margin))

    return (within_reach & is_closing & partially_closed).float()


def gripper_close_when_near_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.05,
    gripper_joint_name: str = "Gripper",
    gripper_close_command: float = 0.5,
    gripper_open_command: float = 0.0,
    gripper_term_name: str = "gripper_action",
) -> torch.Tensor:
    """Reward for gripper closing when near object.
    
    This encourages the agent to close the gripper when the end-effector is close to the object,
    regardless of action state, to help overcome the action bias issue.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Check if end-effector is close to object
    obj_pos = obj.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
    is_near = distance < distance_threshold

    # Get gripper action
    try:
        gripper_term = env.action_manager.get_term(gripper_term_name)
        raw_action = gripper_term.raw_actions.squeeze(-1)
        # Reward negative actions (closing) when near object
        # Use sigmoid to make it smooth: sigmoid(-action) is high when action is negative
        close_action_reward = torch.sigmoid(-raw_action * 3.0)  # Multiply by 3 for stronger gradient
    except Exception:
        # Fallback: no reward if can't access actions
        close_action_reward = torch.zeros_like(distance)

    return (is_near * close_action_reward).float()


def object_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.03,
    gripper_close_threshold: float = 0.6,
) -> torch.Tensor:
    """
    Reward function for detecting if the object is being grasped (robotis version).

    Combines end-effector proximity and gripper closure conditions.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Compute the distance between end-effector and object
    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    # Check if gripper joints are closed beyond threshold
    # For SO-ARM100, we only have one gripper joint
    gripper_joint_pos = robot.data.joint_pos[:, -1]  # Last joint is gripper
    gripper_closed = gripper_joint_pos >= gripper_close_threshold

    # Return reward if both conditions are met
    return (pose_diff <= diff_threshold) & gripper_closed


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

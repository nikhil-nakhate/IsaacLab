from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_joint_position_on_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position: float,
    velocity: float = 0.0,
    set_targets: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set specified joints to a given position (and velocity) at reset.

    Joints are selected via ``asset_cfg.joint_ids`` / ``asset_cfg.joint_names``.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    num_envs = len(env_ids)
    num_j = len(range(robot.num_joints)[joint_ids]) if isinstance(joint_ids, slice) else len(joint_ids)
    pos = torch.full((num_envs, num_j), float(position), device=robot.device)
    vel = torch.full((num_envs, num_j), float(velocity), device=robot.device)

    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids, joint_ids=joint_ids)
    if set_targets:
        robot.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=env_ids)
        robot.set_joint_velocity_target(vel, joint_ids=joint_ids, env_ids=env_ids)


def auto_close_gripper_on_proximity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    diff_threshold: float,
    close_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Automatically command the gripper closed when EE is near the object.

    Runs periodically (e.g., with an "interval" event). It sets joint position targets for the
    specified joints to ``close_value`` for environments where the EE-object distance is below
    ``diff_threshold``.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # resolve env ids
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
    sel = env_ids[dist[env_ids] < diff_threshold]
    if len(sel) == 0:
        return

    joint_ids = asset_cfg.joint_ids
    num_j = len(range(robot.num_joints)[joint_ids]) if isinstance(joint_ids, slice) else len(joint_ids)
    pos = torch.full((len(sel), num_j), float(close_value), device=robot.device)
    vel = torch.zeros((len(sel), num_j), device=robot.device)
    # write directly for immediate effect, and set targets to maintain it
    robot.write_joint_state_to_sim(pos, vel, env_ids=sel, joint_ids=joint_ids)
    robot.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=sel)


def auto_open_gripper_when_far(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    distance_threshold: float,
    open_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Automatically command the gripper open when EE is far from the object.

    Runs periodically (e.g., with an "interval" event). It sets joint position targets for the
    specified joints to ``open_value`` for environments where the EE-object distance is above
    ``distance_threshold``. This helps ensure the gripper remains open on approach.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # resolve env ids
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
    sel = env_ids[dist[env_ids] > distance_threshold]
    if len(sel) == 0:
        return

    joint_ids = asset_cfg.joint_ids
    num_j = len(range(robot.num_joints)[joint_ids]) if isinstance(joint_ids, slice) else len(joint_ids)
    pos = torch.full((len(sel), num_j), float(open_value), device=robot.device)
    vel = torch.zeros((len(sel), num_j), device=robot.device)
    robot.write_joint_state_to_sim(pos, vel, env_ids=sel, joint_ids=joint_ids)
    robot.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=sel)

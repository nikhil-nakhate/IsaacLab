# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg


@configclass
class SoArm100CubeLiftIKRelEnvCfg(joint_pos_env_cfg.SoArm100CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent (sets robot, object, ee_frame, rewards, events, etc.)
        super().__post_init__()

        # Override arm action to use Differential IK (relative pose control)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            body_name="Fixed_Gripper",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            # Align TCP with EE frame (URDF-based heuristic toward pinch center at tips)
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[-0.01, -0.09, 0.0]),
        )


@configclass
class SoArm100CubeLiftIKRelEnvCfg_PLAY(SoArm100CubeLiftIKRelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene and no corruption for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

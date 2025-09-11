# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg, CurriculumTermCfg as CurrTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.common import ViewerCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab_assets.robots.so_arm100 import SO_ARM100_CFG

##
# Pre-defined configs
##


@configclass
class SoArm100CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"], 
            scale=0.3, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Gripper"],
            # For SO-ARM100, lower value closes the jaw, higher opens
            open_command_expr={"Gripper": 0.5},
            close_command_expr={"Gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["Fixed_Gripper"]

        # Set Cube as object - moved closer to robot for easier reach
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(density=3.0),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Fixed_Gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.02, 0.0, 0.1],
                    ),
                ),
            ],
        )

        # Camera configuration for better robot arm view
        self.viewer = ViewerCfg(
            eye=(1.2, -1.2, 0.8),  # Camera position - good overview of robot
            lookat=(0.2, 0.0, 0.3),  # Look at robot arm area
            origin_type="env",  # Use environment origin
            env_index=0,  # Focus on first environment
        )

        # Set a safe pre-grasp posture that places the EE well above the table
        # (approx. ~10 cm depending on USD axes). These values are known-stable
        # from prior config and avoid self-collision.
        self.events.pregrasp_shoulder_pitch = EventTerm(
            func=mdp.set_joint_position_on_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["Shoulder_Pitch"]),
                "position": 0.35,
                "velocity": 0.0,
                "set_targets": True,
            },
        )
        self.events.pregrasp_elbow = EventTerm(
            func=mdp.set_joint_position_on_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["Elbow"]),
                "position": 0.85,
                "velocity": 0.0,
                "set_targets": True,
            },
        )
        self.events.pregrasp_wrist_pitch = EventTerm(
            func=mdp.set_joint_position_on_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["Wrist_Pitch"]),
                "position": -0.75,
                "velocity": 0.0,
                "set_targets": True,
            },
        )
        self.events.pregrasp_wrist_roll = EventTerm(
            func=mdp.set_joint_position_on_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["Wrist_Roll"]),
                "position": 0.0,
                "velocity": 0.0,
                "set_targets": True,
            },
        )
        self.events.open_gripper_on_reset = EventTerm(
            func=mdp.set_joint_position_on_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["Gripper"]),
                "position": 0.5,
                "velocity": 0.0,
                "set_targets": True,
            },
        )

        # # -----------------------
        # # Curriculum: Reach → Lift
        # # Start with strong reaching-only signals; enable lift + grasp later.
        # # -----------------------
        # # Initial reward weights (reaching high, others off)
        # self.rewards.reaching_object.weight = 3.0
        # # keep gripper open on approach initially
        # if hasattr(self.rewards, "gripper_open"):
        #     self.rewards.gripper_open.weight = 3.0
        # self.rewards.lifting_object.weight = 0.0
        # self.rewards.object_grasped.weight = 0.0
        # # Defer goal tracking until after lift behavior emerges
        # if hasattr(self.rewards, "object_goal_tracking"):
        #     self.rewards.object_goal_tracking.weight = 0.0
        # if hasattr(self.rewards, "object_goal_tracking_fine_grained"):
        #     self.rewards.object_goal_tracking_fine_grained.weight = 0.0

        # # After 10k steps, emphasize lifting and reduce reach shaping
        # self.curriculum.enable_lifting = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={"term_name": "lifting_object", "weight": 15.0, "num_steps": 10000},
        # )
        # self.curriculum.enable_grasp = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={"term_name": "object_grasped", "weight": 5.0, "num_steps": 10000},
        # )
        # self.curriculum.reduce_reach_weight = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={"term_name": "reaching_object", "weight": 0.5, "num_steps": 10000},
        # )
        # self.curriculum.gripper_open = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={"term_name": "gripper_open", "weight": 0.0, "num_steps": 10000},
        # )
        # self.curriculum.enable_gripper_close_near = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={"term_name": "gripper_close_near", "weight": 1.0, "num_steps": 10000},
        # )
        # # Bring in goal tracking later to guide post-lift placement
        # if hasattr(self.rewards, "object_goal_tracking"):
        #     self.curriculum.enable_goal_tracking = CurrTerm(
        #         func=mdp.modify_reward_weight,
        #         params={"term_name": "object_goal_tracking", "weight": 8.0, "num_steps": 12000},
        #     )
        # if hasattr(self.rewards, "object_goal_tracking_fine_grained"):
        #     self.curriculum.enable_goal_tracking_fine = CurrTerm(
        #         func=mdp.modify_reward_weight,
        #         params={"term_name": "object_goal_tracking_fine_grained", "weight": 5.0, "num_steps": 15000},
        #     )

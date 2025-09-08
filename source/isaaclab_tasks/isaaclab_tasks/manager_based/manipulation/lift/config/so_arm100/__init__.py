# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, joint_pos_env_cfg, ik_rel_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Lift-Cube-SO-ARM100-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.SoArm100CubeLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.SoArm100CubeLiftPPORunnerCfg,
    },
    disable_env_checker=True,
)

# Inverse Kinematics - Relative Pose Control
gym.register(
    id="Isaac-Lift-Cube-SO-ARM100-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.SoArm100CubeLiftIKRelEnvCfg,
        # Reuse the same PPO defaults; IK tracking is handled by the controller
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.SoArm100CubeLiftPPORunnerCfg,
    },
    disable_env_checker=True,
)


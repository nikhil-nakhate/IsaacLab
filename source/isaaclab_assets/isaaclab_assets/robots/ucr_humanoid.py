# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UCR Humanoid (external USD imported into assets).

USD Source (copied into repo): data/Robots/ucr_humanoid/v0H.usd
"""

from __future__ import annotations

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Assets root relative to this package
_ASSETS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

UCR_HUMANOID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSETS_ROOT}/Robots/ucr_humanoid/v0H.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Generic actuator covering all joints; tune per-joint if needed later
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=30.0,
            damping=3.0,
            effort_limit_sim=250.0,
            velocity_limit_sim=60.0,
        ),
    },
)
"""Configuration for the UCR humanoid USD imported into IsaacLab assets."""


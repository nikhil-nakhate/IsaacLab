from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
import os

"""Configuration for the Table with Cube Scene"""
# Define leisaac path explicitly since we're running from IsaacLab directory
leisaac_path = "/home/nikhil/Code/leisaac"
ASSETS_ROOT = Path(os.path.join(leisaac_path, 'assets'))
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"
TABLE_WITH_CUBE_USD_PATH = str(SCENES_ROOT / "table_with_cube" / "scene.usd")

TABLE_WITH_CUBE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TABLE_WITH_CUBE_USD_PATH,
    )
)

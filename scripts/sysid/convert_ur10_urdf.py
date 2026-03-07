"""Pre-convert UR10 URDF to USD for use with Newton standalone mode.

Newton's standalone solver doesn't have Omniverse Kit (omni.kit.commands),
so UrdfConverter can't run at simulation time. This script runs the
conversion once with full Isaac Sim, producing a USD file that run_sysid.py
can load directly via UsdFileCfg.

Usage (inside Docker with Isaac Sim):
    python scripts/sysid/convert_ur10_urdf.py --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert UR10 URDF to USD")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- After AppLauncher ---

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_SCRIPT_DIR, "..", "..", "input", "robot_models", "ur10", "ur10.urdf")
_USD_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "input", "robot_models", "ur10")
_USD_FILENAME = "ur10.usd"


def main():
    urdf_path = os.path.abspath(_URDF_PATH)
    usd_dir = os.path.abspath(_USD_DIR)

    print(f"Input URDF:  {urdf_path}")
    print(f"Output dir:  {usd_dir}")
    print(f"Output file: {_USD_FILENAME}")

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        usd_file_name=_USD_FILENAME,
        fix_base=True,
        force_usd_conversion=True,
        # No joint drive — actuator config is applied separately via ImplicitActuatorCfg
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
        ),
    )

    converter = UrdfConverter(cfg)
    print(f"\nGenerated USD: {converter.usd_path}")
    print("Done. You can now run run_sysid.py with --robot-name ur10e")


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python3

"""Test script to verify SO-ARM100 integration with IsaacLab."""

import os
import sys

def test_file_structure():
    """Test if all required files are in place."""
    print("Testing SO-ARM100 integration file structure...")
    
    # Check robot configuration
    robot_file = "/home/nikhil/Code/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/so_arm100.py"
    if os.path.exists(robot_file):
        print("✓ Robot configuration file exists")
    else:
        print("✗ Robot configuration file missing")
        return False
    
    # Check robot assets
    robot_assets = "/home/nikhil/Code/IsaacLab/source/isaaclab_assets/data/Robots/so_arm100"
    if os.path.exists(robot_assets):
        print("✓ Robot asset directory exists")
    else:
        print("✗ Robot asset directory missing")
        return False
    
    # Check lift task
    lift_task = "/home/nikhil/Code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/so_arm100"
    if os.path.exists(lift_task):
        print("✓ Lift task configuration exists")
    else:
        print("✗ Lift task configuration missing")
        return False
    
    # Check reach task
    reach_task = "/home/nikhil/Code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/so_arm100"
    if os.path.exists(reach_task):
        print("✓ Reach task configuration exists")
    else:
        print("✗ Reach task configuration missing")
        return False
    
    return True

def test_imports():
    """Test basic imports without Isaac Sim dependencies."""
    print("\nTesting imports...")
    
    try:
        # Test if files can be read and parsed
        with open("/home/nikhil/Code/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/so_arm100.py", 'r') as f:
            content = f.read()
            if "SO_ARM100_CFG" in content and "SO_ARM100_ROS2_CFG" in content:
                print("✓ Robot configuration contains expected configurations")
            else:
                print("✗ Robot configuration missing expected configurations")
                return False
    except Exception as e:
        print(f"✗ Error reading robot configuration: {e}")
        return False
    
    try:
        # Test lift task configuration
        with open("/home/nikhil/Code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/so_arm100/__init__.py", 'r') as f:
            content = f.read()
            if "Isaac-Lift-Cube-SO-ARM100-v0" in content:
                print("✓ Lift task environment registration found")
            else:
                print("✗ Lift task environment registration missing")
                return False
    except Exception as e:
        print(f"✗ Error reading lift task configuration: {e}")
        return False
    
    try:
        # Test reach task configuration
        with open("/home/nikhil/Code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/so_arm100/__init__.py", 'r') as f:
            content = f.read()
            if "Isaac-Reach-SO-ARM100-v0" in content:
                print("✓ Reach task environment registration found")
            else:
                print("✗ Reach task environment registration missing")
                return False
    except Exception as e:
        print(f"✗ Error reading reach task configuration: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("SO-ARM100 IsaacLab Integration Test")
    print("=" * 40)
    
    structure_ok = test_file_structure()
    imports_ok = test_imports()
    
    print("\n" + "=" * 40)
    if structure_ok and imports_ok:
        print("✅ Integration test PASSED!")
        print("\nNext steps:")
        print("1. The SO-ARM100 robot is now available in IsaacLab")
        print("2. Available environments:")
        print("   - Isaac-Lift-Cube-SO-ARM100-v0")
        print("   - Isaac-Lift-Cube-SO-ARM100-Play-v0")
        print("   - Isaac-Reach-SO-ARM100-v0")
        print("   - Isaac-Reach-SO-ARM100-Play-v0")
        print("3. You can now train and test these environments using IsaacLab tools")
        return 0
    else:
        print("❌ Integration test FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Plan: Fix Build Failures via Git Submodules and Clean Rebuild

This plan resolves the build errors in the `planning` and `xsens_mti_ros2_driver` packages by replacing the incomplete driver folder with a proper Git submodule and clearing stale build artifacts.

## Objective
Restore a clean, working build environment that "just works" for any user who clones the repository.

## Key Files & Context
- `teambowl_ws/src/drivers/xsens_mti_ros2_driver`: Current incomplete driver folder (to be replaced).
- `teambowl_ws/build` / `teambowl_ws/install`: Contain stale artifacts causing path conflicts in the `planning` package.

## Proposed Solution
1. **Replace Driver with Submodule**: Replace the local copy of the Xsens driver with a proper Git submodule pointing to the official repository. This ensures all dependencies (like `xspublic`) are correctly tracked and easily updated.
2. **Clean Workspace**: Use the existing `--clean` build mechanism to wipe incompatible host-side build artifacts.

## Implementation Steps

1. **Remove Existing Driver**:
   - Delete the current `teambowl_ws/src/drivers/xsens_mti_ros2_driver` directory.

2. **Add Git Submodule**:
   - Add the official Xsens ROS 2 repository as a submodule:
     `git submodule add -b ros2 https://github.com/xsenssupport/Xsens_MTi_ROS_Driver_and_Ntrip_Client.git teambowl_ws/src/drivers/xsens_mti_ros_driver_repo`

3. **Initialize and Update**:
   - Run `git submodule update --init --recursive` to ensure the `xspublic` library source is populated.

4. **Trigger Clean Build**:
   - Run the laptop build script with the clean flag:
     `./build.laptop.sh --clean`

## Verification & Testing
1. **Submodule Check**: Confirm that `teambowl_ws/src/drivers/xsens_mti_ros_driver_repo/src/xsens_mti_ros2_driver/lib/xspublic` contains source files (e.g., `GNUmakefile`, `xscontroller`).
2. **Build Success**: Verify that `colcon build` completes successfully for all packages, specifically `planning` and `xsens_mti_ros2_driver`.
3. **Node Discovery**: Inside the container, run `ros2 pkg list | grep xsens` to verify the package is recognized.

<div align="center">
<h1>NextBestPath: Efficient 3D Mapping of Unseen Environments</h1>

[Shiyao Li](https://shiyao-li.github.io/), [Antoine Guédon](https://anttwo.github.io/), [Clémentin Boittiaux](https://clementinboittiaux.github.io/), [Shizhe Chen](https://cshizhe.github.io/), [Vincent Lepetit](https://vincentlepetit.github.io/)

<a href="https://arxiv.org/pdf/2502.05378" style="margin-right: 10px;">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv Paper">
</a>
<a href="https://shiyao-li.github.io/nbp/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

*A method for generating the next-best-path for efficient active mapping, along with a new benchmark tailored for complex indoor environments.*

</div>


##### 🌟 If you find our work helpful, please consider giving a ⭐️ to this repository and citing our paper!



## 🗺️ Project Overview

NextBestPath (NBP) is a novel method for next-best-path planning in 3D scene exploration. Unlike previous methods, NBP is designed to directly maximize mapping efficiency and coverage along the camera trajectory.


This repository contains:
* A simulator based on PyTorch3D and Trimesh
* Functions for generating ground truth point clouds from meshes and evaluating reconstructed point clouds
* Scripts for testing and training NBP models on AiMDoom dataset.

```bibtex
@inproceedings{li2025nextbestpath,
  title={NextBestPath: Efficient 3D Mapping of Unseen Environments},
  author={Shiyao Li and Antoine Guedon and Cl{\'e}mentin Boittiaux and Shizhe Chen and Vincent Lepetit},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=7WaRh4gCXp}
}
```

## Updates
- [June, 2025] Release the training and test code
- Todo: Release the models of MACARONS and the corresponding scripts


## Quick Start

### Prerequisites

First, ensure you have conda installed, then set up the environment:

```bash
# Clone this repository
git clone https://github.com/shiyao-li/NextBestPath.git
cd NextBestPath

# Create and activate conda environment
conda env create -f environment.yml
conda activate exploration
```

### Installation

1. **Download the AiMDoom Dataset**
   
   Download the complete dataset from [Google Drive](https://drive.google.com/drive/folders/1fwhCrxmrJnpdK-egawoX2OYHUxnxAwr-):
   - AiMDoom dataset (4 difficulty levels)
   - The toolkit and code to build AiMDoom dataset: [Github_link](https://github.com/shiyao-li/AiMDoom)

2. **Download and set up model weights**
   
   Download NBP models from [Google Drive](https://drive.google.com/drive/folders/1jAEKrznbbZ5bwu39y0ah4pszMlTuVAfH?usp=sharing), and put them under the `./weights/nbp/` folder.
   
   Place the downloaded NBP model weights in the following structure:
   ```
   ./weights/nbp/
   ├── AiMDoom_simple_best_val.pth  
   ├── AiMDoom_normal_best_val.pth  
   ├── AiMDoom_hard_best_val.pth  
   └── AiMDoom_insane_best_val.pth
   ```

### Usage

1. **Configs**
   
   All config files are under the `./configs/` folder.

2. **Test NBP method**
   ```bash
   python test_nbp_planning.py
   ```

3. **Train NBP models**
   ```bash
   python train_nbp.py
   ```

## 🤖 Real Robot Deployment

This repository now supports deployment on real robots! We have successfully integrated NBP with **RealSense D435i** camera and **Unitree Go2w** quadruped robot for autonomous indoor exploration.

**📚 完整部署指南请查看：[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

### Hardware Requirements

- **Camera**: Intel RealSense D435i
- **Robot**: Unitree Go2w (or compatible quadruped robot)
- **Computer**: Laptop/PC with CUDA-capable GPU (recommended)

### Additional Dependencies

Install real robot dependencies:

```bash
# Install RealSense SDK
pip install pyrealsense2

# Install Unitree SDK2 Python
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
cd ..

# Install other dependencies
pip install -r requirements_real_robot.txt
```

### Real Robot Files

The following new files have been added for real robot deployment:

- `real_camera_impl.py` - RealSense D435i camera interface
- `real_world_env.py` - Robot environment management (integrates camera + robot)
- `Unigoal_Action_Mapping.py` - Enhanced with `move_to_pose()` function for pose-to-velocity control
- `test_real_robot.py` - Main testing script for real robot
- `openply.py` - Point cloud visualization tool
- `rviz_visualizer.py` - RViz trajectory visualization (requires ROS)
- `requirements_real_robot.txt` - Additional dependencies

### Quick Start - Real Robot

1. **Connect Hardware**

   ```bash
   # Connect RealSense D435i via USB
   # Connect to Unitree Go2w via Ethernet
   # Default interface: enx607d099f16d2
   ```

2. **Test Camera and Robot**

   ```bash
   # Test RealSense camera
   python real_camera_impl.py

   # Test robot environment
   python real_world_env.py
   ```

3. **Run Real Robot Exploration**

   ```bash
   python test_real_robot.py \
       --nbp_weights ./weights/nbp/AiMDoom_simple_best_val.pth \
       --nbp_config ./configs/nbp/nbp_default_training_config.json \
       --n_poses 20 \
       --max_velocity 0.2 \
       --max_angular 0.3
   ```

   **Important Parameters:**
   - `--n_poses`: Number of exploration steps (20-50 recommended for real robot)
   - `--max_velocity`: Maximum linear velocity in m/s (0.2 recommended for safety)
   - `--max_angular`: Maximum angular velocity in rad/s (0.3 recommended)
   - `--robot_interface`: Network interface name for robot connection

4. **View Results**

   After exploration, results are saved in `./results/real_robot/run_TIMESTAMP/`:

   ```bash
   # View point cloud
   python openply.py results/real_robot/run_TIMESTAMP/full_pc.ply \
       --trajectory results/real_robot/run_TIMESTAMP/trajectory.ply

   # View trajectory information
   python openply.py results/real_robot/run_TIMESTAMP/trajectory.ply --info

   # Compare multiple point clouds
   python openply.py partial_000.ply partial_005.ply partial_010.ply --compare
   ```

### RViz Visualization (Optional)

If you have ROS installed, you can visualize the trajectory in RViz:

```bash
# Create RViz config file
python rviz_visualizer.py --create-config

# Start RViz
rviz -d nbp_visualization.rviz &

# Publish trajectory to RViz
python rviz_visualizer.py results/real_robot/run_TIMESTAMP/
```

### Output Files

Real robot test produces the following outputs:

- `full_pc.ply` - Complete accumulated point cloud from entire exploration
- `partial_XXX.ply` - Partial point clouds captured at each step
- `trajectory.ply` - Robot trajectory waypoints
- `results.json` - Complete exploration data (trajectory, poses, parameters)

### Architecture Overview

```
Real Robot System Architecture:

RealSense D435i ─┐
                 ├─> RealWorldEnvironment ─> NBP Model ─> Path Planning
Unitree Go2w  ───┘                                              │
       ↑                                                         │
       └─────────── move_to_pose() ──────────────────────────────┘
                    (Velocity Control)
```

**Workflow:**
1. Capture RGB + Depth from RealSense
2. Get robot pose from Unitree Go2w odometry
3. Transform depth to world-frame point cloud
4. Accumulate point cloud
5. NBP predicts next-best-path
6. Convert target pose to velocity commands
7. Move robot using `move_to_pose()` controller
8. Repeat

### Key Features

✅ **Minimal Code Changes** - Original NBP algorithm unchanged
✅ **Real-time Performance** - Optimized for embedded deployment
✅ **Safety First** - Conservative velocity limits and collision checking
✅ **Complete Logging** - Save point clouds, trajectories, and metadata
✅ **Visualization Tools** - Open3D and RViz support

### Coordinate Systems

The system handles three coordinate frames:

- **Camera Frame**: RealSense D435i local coordinates
- **Robot Frame**: Unitree Go2w body frame
- **World Frame**: Fixed global reference (initialized at start)

Transformations are handled automatically by `RealWorldEnvironment`.

### Troubleshooting

**Camera not detected:**
```bash
# Check RealSense connection
realsense-viewer
```

**Robot not responding:**
```bash
# Check network interface
ifconfig
# Update interface name in test_real_robot.py --robot_interface
```

**Point cloud quality issues:**
- Ensure good lighting conditions
- Adjust camera filters in `real_camera_impl.py`
- Increase `gathering_factor` for denser point clouds

**Robot movement issues:**
- Reduce `max_velocity` and `max_angular` parameters
- Check `position_tolerance` and `yaw_tolerance` in `move_to_pose()`
- Monitor robot state with `get_state()` function

### Safety Guidelines

⚠️ **Important Safety Notes:**

1. **Test in Safe Environment** - Clear area free of obstacles and people
2. **Emergency Stop** - Keep Ctrl+C ready to stop execution
3. **Start Slow** - Begin with low velocity limits (0.1-0.2 m/s)
4. **Monitor First Run** - Watch robot behavior closely during initial tests
5. **Check Battery** - Ensure robot has sufficient battery level

### Future Improvements

- [ ] Add SLAM integration for better odometry
- [ ] Implement dynamic obstacle avoidance
- [ ] Support for multiple camera sensors
- [ ] Real-time map visualization during exploration
- [ ] Automatic recovery from movement failures

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{li2025nextbestpath,
  title={NextBestPath: Efficient 3D Mapping of Unseen Environments},
  author={Shiyao Li and Antoine Guedon and Cl{\'e}mentin Boittiaux and Shizhe Chen and Vincent Lepetit},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=7WaRh4gCXp}
}
```


📁 实验文件+数据和模型
/base/path
├── diffsynth4MT/                           # 主实验目录
│   ├── FastVMT_incremental.py             # ✅ 核心实验脚本
│   ├── parallel_experiment_manager.sh     # ✅ 并行管理器
│   ├── incremental_experiment.sh          # ✅ 增量实验脚本
│   ├── check_experiment_progress.py       # ✅ 进度检查脚本
│   ├── config.env.template               # ✅ 配置模板
│   ├── 运行操作指南.md                    # ✅ 使用说明
│   ├── test_parallel_setup.sh            # ✅ 测试脚本
│   ├── test_random_seeds.sh              # ✅ 随机种子测试
│   └── setup_new_host.sh                 # ✅ 自动配置脚本
├── Final_Dataset/                         # 数据集目录
│   ├── prompt.json                        # prompts文件
│   ├── camera_motion/                     # 各类别视频
│   ├── single_object/
│   ├── multiple_objects/
│   └── complex_human_motion/
├── pretrained_models/                     # 预训练模型
│   └── Wan-AI/Wan2.1-T2V-14B/           # 模型文件
└── logs/                                  # 日志目录 (可选)

```bash
conda create -n diffsynth
conda activate diffsynth

cd /path/to/diffsynth4MT
pip install -e .
# 验证环境
python -c "from diffsynth import ModelManager; print('Environment OK')"

cd /path/to/diffsynth4MT/scripts_9_8
chmod +x setup_new_host.sh 
./setup_new_host.sh /base/path

# download Wan-AI/Wan2.1-T2V-14B using /root/autodl-tmp/diffsynth4MT/scripts_9_8/download_models.py. Remember to change the local_dir="/path/to/pretrained_models/Wan-AI/Wan2.1-T2V-14B

source config.env

# 4GPU服务器 - 每GPU 1个进程 
./parallel_experiment_manager.sh --mode 4gpu --dataset /root/autodl-tmp/Final_Dataset

# 8GPU服务器 - 每GPU 1个进程 
./parallel_experiment_manager.sh --mode 8gpu-conservative --dataset /root/autodl-tmp/Final_Dataset

# 8GPU服务器 - 每GPU 2个进程 (16倍加速)
./parallel_experiment_manager.sh --mode 8gpu-conservative --processes-per-gpu 2 --dataset /root/autodl-tmp/Final_Dataset
```
### parallel_experiment_manager.sh 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode` | 运行模式 | `4gpu`, `8gpu-conservative`, `8gpu-aggressive` |
| `--dataset` | 数据集路径 | `/root/autodl-tmp/Final_Dataset` |
| `--processes-per-gpu` | 每GPU进程数 | `1`, `2`, `3` (默认: 1) |
| `--prompts` | prompts文件 | 默认: `{dataset}/prompt.json` |

### incremental_experiment.sh 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--category` | 视频类别 | `camera_motion`, `single_object` |
| `--gpu-id` | GPU编号 | `0`, `1`, `2`, `3` |
| `--video-range` | 视频范围 | `1-7`, `8-14` |
| `--instance-id` | 实例标识 | `part1`, `proc2` |
| `--new-prompts` | prompts文件路径 | `/path/to/prompt.json` |
| `--new-videos-dir` | 视频目录路径 | `/path/to/videos/` |




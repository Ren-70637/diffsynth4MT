# Wan2.1-T2V-14B 多GPU分布式实验使用指南

## 快速开始

### 1. 一键启动实验
```bash
cd /root/autodl-tmp
chmod +x quick_start.sh
./quick_start.sh
```

### 2. 监控实验进度
```bash
# 进入监控界面（推荐）
tmux attach -t wan_monitor

# 或者查看当前进度
python check_experiment.py

# 详细状态报告
python check_experiment.py --detailed
```

### 3. 管理实验
```bash
# 查看所有会话
tmux list-sessions

# 查看特定GPU进程
tmux attach -t wan_gpu0_proc0

# 停止实验
python run_experiment.py --action stop
```

## 文件结构

```
/root/autodl-tmp/
├── experiment_utils.py         # 实验工具函数
├── gpu_worker.py              # GPU工作进程
├── run_experiment.py          # 主控脚本
├── launch_experiment.sh       # tmux启动脚本
├── quick_start.sh            # 快速启动脚本
├── check_experiment.py       # 状态检查脚本
├── EXPERIMENT_SETUP_GUIDE.md # 详细操作指南
└── experiment_results/       # 实验结果目录
    ├── camera_motion/
    ├── single_object/
    ├── multiple_objects/
    ├── complex_human_motion/
    └── logs/
```

## 实验配置

- **GPU分配**:
  - GPU 0: camera_motion (14个视频)
  - GPU 1: single_object (33个视频)
  - GPU 2: multiple_objects (13个视频)
  - GPU 3: complex_human_motion (8个视频)

- **每GPU进程数**: 2个并行进程
- **每视频**: 5个prompt × 2个seed = 10个生成视频
- **总计**: 680个视频

## 常用命令

### 实验控制
```bash
# 启动实验
python run_experiment.py --action start

# 停止实验
python run_experiment.py --action stop

# 查看实验概要
python run_experiment.py --action summary

# 查看进度
python run_experiment.py --action progress
```

### tmux会话管理
```bash
# 查看所有会话
tmux list-sessions

# 进入主监控界面
tmux attach -t wan_monitor

# 进入特定GPU进程（示例）
tmux attach -t wan_gpu0_proc0

# 创建新窗口
tmux new-window -t wan_monitor

# 分离会话（后台运行）
Ctrl+B, D
```

### 状态监控
```bash
# 简单状态
python check_experiment.py

# 详细报告
python check_experiment.py --detailed

# 实时GPU监控
watch -n 1 nvidia-smi

# 查看特定日志
tail -f experiment_results/logs/gpu0_process0.log
```

## 故障排除

### 常见问题

1. **显存不足**
   ```bash
   # 检查显存使用
   nvidia-smi
   
   # 减少并行进程数
   python run_experiment.py --action start --processes_per_gpu 1
   ```

2. **模型加载失败**
   ```bash
   # 检查模型文件
   ls -la /root/autodl-tmp/Wan-AI/Wan2.1-T2V-14B/
   
   # 查看错误日志
   tail -f experiment_results/logs/gpu0_process0.log
   ```

3. **tmux会话异常**
   ```bash
   # 清理所有会话
   python run_experiment.py --action stop
   
   # 重新启动
   ./quick_start.sh
   ```

4. **进程卡死**
   ```bash
   # 查看进程状态
   python check_experiment.py --detailed
   
   # 重启特定GPU的进程
   tmux kill-session -t wan_gpu0_proc0
   tmux new-session -d -s wan_gpu0_proc0 "cd /root/autodl-tmp && conda activate diffsynth && python gpu_worker.py --gpu_id 0 --process_id 0"
   ```

### 恢复策略

- **断电重启**: 重新运行 `./quick_start.sh`，脚本会自动跳过已完成的任务
- **部分失败**: 查看日志定位问题，重启相关进程
- **存储不足**: 清理中间文件，增加存储空间

## 性能优化

### 显存优化
- 使用 `torch.bfloat16` 减少显存占用
- 启用 `tiled=True` 进行分块推理
- 设置合适的 `num_persistent_param_in_dit`

### 并行优化
- 根据显存情况调整每GPU进程数
- 避免同时启动过多进程
- 使用任务队列平衡负载

### 存储优化
- 定期清理临时文件
- 压缩完成的视频文件
- 监控磁盘空间使用

## 实验数据

### 输出格式
每个生成的视频按以下格式命名：
```
experiment_results/{category}/{video_name}/seed_{seed}_prompt_{prompt_idx}.mp4
```

### 质量设置
- 分辨率: 832×480
- 帧数: 81帧
- 帧率: 15 FPS
- 推理步数: 50步
- 质量等级: 5

### 预期结果
- 总视频数: 580个
- 预计大小: ~70GB (81帧×580个视频)
- 完成时间: 8-10小时

## 安全注意事项

1. **备份重要数据**: 实验前备份原始数据
2. **监控系统资源**: 定期检查CPU、内存、存储使用情况
3. **网络稳定性**: 使用tmux确保断网不影响实验
4. **定期检查**: 每2-3小时检查一次实验状态

## 支持和帮助

如果遇到问题，请：

1. 查看详细日志: `python check_experiment.py --detailed`
2. 检查系统资源: `nvidia-smi` 和 `htop`
3. 查看错误信息: `tail -f experiment_results/logs/*.log`
4. 重启相关进程或整个实验

## 实验完成后

```bash
# 检查结果完整性
find experiment_results -name "*.mp4" | wc -l

# 打包结果（可选）
tar -czf experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz experiment_results/

# 清理tmux会话
python run_experiment.py --action stop
```

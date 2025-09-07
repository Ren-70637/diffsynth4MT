# Wan2.1-T2V-14B 多GPU实验 - 最终安装总结

## 🎯 实验配置完成

已成功创建完整的4-GPU分布式实验环境，总计将生成 **580个视频**。

### 📊 实验规模
- **GPU 0 (camera_motion)**: 14个视频 → 140个任务
- **GPU 1 (single_object)**: 24个视频 → 240个任务  
- **GPU 2 (multiple_objects)**: 13个视频 → 130个任务
- **GPU 3 (complex_human_motion)**: 7个视频 → 70个任务

### 🔧 已创建的脚本文件

1. **`experiment_utils.py`** - 核心工具函数库
2. **`gpu_worker.py`** - 单GPU工作进程脚本
3. **`run_experiment.py`** - 主控制脚本
4. **`launch_experiment.sh`** - tmux启动脚本
5. **`quick_start.sh`** - 快速启动脚本
6. **`check_experiment.py`** - 状态检查脚本

### 📚 文档文件

1. **`EXPERIMENT_SETUP_GUIDE.md`** - 详细操作指南
2. **`README_EXPERIMENT.md`** - 使用说明书
3. **`FINAL_SETUP_SUMMARY.md`** - 本总结文档

## 🚀 启动命令

### 方法1: 快速启动（推荐）
```bash
cd /root/autodl-tmp
./quick_start.sh
```

### 方法2: 分步启动
```bash
cd /root/autodl-tmp
# 查看实验概要
python run_experiment.py --action summary

# 启动实验
./launch_experiment.sh
```

## 📈 监控命令

```bash
# 进入监控界面（推荐）
tmux attach -t wan_monitor

# 查看当前进度
python check_experiment.py

# 详细状态报告
python check_experiment.py --detailed

# 查看GPU状态
watch -n 1 nvidia-smi
```

## 🛠️ 管理命令

```bash
# 查看所有tmux会话
tmux list-sessions

# 查看特定GPU进程
tmux attach -t wan_gpu0_proc0

# 停止实验
python run_experiment.py --action stop

# 查看日志
tail -f experiment_results/logs/gpu0_process0.log
```

## ⚡ 性能配置

- **并行度**: 每GPU 2个进程，总计8个并行进程
- **显存优化**: 使用bfloat16和tiled推理
- **容错机制**: 自动跳过已完成任务，支持断点续传
- **输出格式**: 832×480, 81帧, 15fps
- **预计时间**: 8-10小时完成全部580个视频

## 🗂️ 输出结构

```
experiment_results/
├── camera_motion/
│   ├── ref_镜头仰角旋转_832x480/
│   │   ├── seed_42_prompt_0.mp4
│   │   ├── seed_42_prompt_1.mp4
│   │   └── ...
│   └── ...
├── single_object/
├── multiple_objects/
├── complex_human_motion/
└── logs/
    ├── gpu0_process0.log
    └── ...
```

## ✅ 预启动检查清单

在启动实验前，请确认：

- [ ] 已激活diffsynth conda环境
- [ ] 4张GPU都可用且显存充足
- [ ] 模型文件完整（约56GB）
- [ ] 参考视频文件完整
- [ ] 磁盘空间充足（建议100GB+）
- [ ] tmux已安装并可用

## 🚨 重要提醒

1. **使用tmux**: 确保实验在后台运行，断网不影响
2. **定期检查**: 建议每2-3小时检查一次进度
3. **显存监控**: 注意GPU显存使用，避免OOM
4. **日志查看**: 出现问题时及时查看日志文件
5. **备份数据**: 实验开始前备份重要数据

## 🎉 启动实验

现在可以运行以下命令启动实验：

```bash
cd /root/autodl-tmp
./quick_start.sh
```

实验将在后台自动运行，预计8-10小时后完成所有580个视频的生成！

---

*所有脚本已准备就绪，祝实验顺利！* 🎬✨

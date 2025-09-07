# 4-GPU分布式Wan2.1-T2V-14B视频生成实验操作指南

## 实验概述

本实验将使用4张NVIDIA A800 GPU并行运行Wan2.1-T2V-14B模型，基于参考视频进行运动迁移生成实验。每个GPU负责处理一个视频类别，使用多进程并行以最大化GPU利用率。

## 环境信息
- **GPU数量**: 4张 NVIDIA A800 80GB PCIe
- **可用显存**: 每张卡约80GB
- **模型**: Wan2.1-T2V-14B (约28GB模型参数)
- **Python环境**: diffsynth虚拟环境
- **会话管理**: tmux (防止断线中断)

## 实验数据分布

### 视频类别分配（每GPU一类）:
- **GPU 0**: `camera_motion` (14个视频)
- **GPU 1**: `single_object` (33个视频)  
- **GPU 2**: `multiple_objects` (13个视频)
- **GPU 3**: `complex_human_motion` (8个视频)

### 每个视频的处理规格:
- **提示词数量**: 每个视频对应5个prompt
- **Seeds**: 每个prompt使用2个不同的seed (42, 142)
- **输出格式**: 832×480分辨率, 81帧, 15fps
- **总生成数量**: 每个视频生成10个视频 (5×2)

### 估算工作量:
- **GPU 0**: 14 × 5 × 2 = 140个视频
- **GPU 1**: 33 × 5 × 2 = 330个视频  
- **GPU 2**: 13 × 5 × 2 = 130个视频
- **GPU 3**: 8 × 5 × 2 = 80个视频
- **总计**: 680个视频

## 并行策略

### 显存分配估算:
- **模型加载**: ~28GB
- **单次推理峰值**: ~35-40GB
- **可用于并行**: 80GB - 40GB = 40GB余量

### 进程分配策略:
- **GPU 0 (camera_motion)**: 2个并行进程
- **GPU 1 (single_object)**: 2个并行进程  
- **GPU 2 (multiple_objects)**: 2个并行进程
- **GPU 3 (complex_human_motion)**: 2个并行进程

## 输出组织结构

```
/root/autodl-tmp/experiment_results/
├── camera_motion/
│   ├── ref_镜头仰角旋转_832x480/
│   │   ├── seed_42_prompt_0.mp4
│   │   ├── seed_42_prompt_1.mp4
│   │   ├── seed_142_prompt_0.mp4
│   │   └── ...
│   └── ...
├── single_object/
├── multiple_objects/
├── complex_human_motion/
└── logs/
    ├── gpu0_process0.log
    ├── gpu0_process1.log
    └── ...
```

## 关键技术要点

### 1. 显存管理
- 使用`pipe.enable_vram_management()`进行显存优化
- 采用`torch.bfloat16`数据类型减少显存占用
- 每个进程独立加载模型，避免进程间干扰

### 2. 任务分配
- 基于视频数量均匀分配GPU负载
- 每个GPU内部使用队列管理任务
- 支持进程失败自动重试机制

### 3. 容错机制
- 每个进程独立运行，单进程失败不影响其他
- 详细日志记录，便于调试
- 支持断点续传，跳过已生成的视频

### 4. 性能优化
- 预加载参考视频到内存
- 批量处理相同参考视频的不同prompt
- 使用tiled推理减少显存峰值

## 预期运行时间

### 单个视频生成时间估算:
- **模型加载**: ~2分钟
- **单次推理**: ~2-4分钟 (50步)
- **视频保存**: ~10秒

### 总体时间估算:
- **最繁重GPU (GPU1)**: 330个视频 ÷ 2进程 = 165个/进程
- **单进程时间**: 165 × 3分钟 = 8.25小时
- **预期总时间**: 约8-10小时 (考虑加载和其他开销)

## 监控和管理

### tmux会话管理:
- **主会话**: `wan_experiment_main`
- **子会话**: `gpu0_proc0`, `gpu0_proc1`, `gpu1_proc0`, `gpu1_proc1`等

### 实时监控:
```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看tmux会话
tmux list-sessions

# 查看特定进程日志
tail -f /root/autodl-tmp/experiment_results/logs/gpu0_process0.log
```

### 进度查看:
```bash
# 查看已生成视频数量
find /root/autodl-tmp/experiment_results -name "*.mp4" | wc -l

# 查看各类别进度
for category in camera_motion single_object multiple_objects complex_human_motion; do
    echo "$category: $(find /root/autodl-tmp/experiment_results/$category -name "*.mp4" 2>/dev/null | wc -l)"
done
```

## 错误处理和恢复

### 常见问题:
1. **显存不足**: 降低并行进程数
2. **模型加载失败**: 检查模型路径和权限
3. **视频读取错误**: 验证参考视频完整性
4. **网络中断**: tmux保证进程继续运行

### 恢复策略:
- 脚本自动跳过已存在的输出文件
- 支持从任意中断点重新启动
- 详细错误日志便于问题定位

## 资源要求验证

### 存储空间:
- **输入数据**: ~500MB (参考视频)
- **输出视频**: 680个 × 50MB ≈ 34GB
- **模型文件**: ~56GB
- **建议预留**: 100GB

### 网络要求:
- 初始模型下载 (如未下载)
- 运行期间无网络依赖

### 系统要求:
- CUDA 12.x 兼容驱动
- Python 3.8+
- 足够的系统内存 (建议64GB+)

## 启动步骤概览

1. **环境激活**: `conda activate diffsynth`
2. **脚本准备**: 运行脚本生成器
3. **tmux启动**: 执行主启动脚本
4. **监控运行**: 检查日志和进度
5. **结果收集**: 验证输出完整性

这个实验设计充分利用了4张A800的计算能力，预计能在8-10小时内完成全部680个视频的生成任务。

# GPU自适应功能说明

## 🚀 更新内容

将原本固定支持4卡的脚本升级为自动适应4卡、8卡或任意数量GPU的系统。

## 📊 支持的GPU配置

- ✅ **4卡系统**: GPU 0-3
- ✅ **8卡系统**: GPU 0-7  
- ✅ **任意数量**: 自动检测系统中的GPU数量

## 🔧 更新的脚本

### 1. incremental_experiment.sh
**新功能**:
- 自动检测系统GPU数量
- 验证指定GPU ID的有效性
- 支持任意数量的GPU

**使用方法**:
```bash
# 自动选择空闲GPU (支持任意数量)
./incremental_experiment.sh \
    --category single_object \
    --new-prompts /path/to/new_prompts.json \
    --new-videos-dir /path/to/new/videos

# 指定GPU ID (会验证有效性)
./incremental_experiment.sh \
    --category single_object \
    --new-prompts /path/to/new_prompts.json \
    --new-videos-dir /path/to/new/videos \
    --gpu-id 7  # 在8卡系统中有效
```

### 2. check_experiment_progress.py
**新功能**:
- 生成的续传脚本也支持自动GPU检测
- 可以处理任意数量的GPU

**使用方法**:
```bash
# 检查进度并生成续传脚本
python check_experiment_progress.py --category single_object --generate_resume
```

## 🧪 测试脚本

### test_gpu_detection.sh
用于验证GPU检测功能是否正常工作。

**使用方法**:
```bash
./test_gpu_detection.sh
```

**测试内容**:
- 检测系统GPU数量
- 显示GPU信息
- 验证GPU范围
- 模拟空闲GPU选择

## ⚡ 实际应用示例

### 4卡系统示例
```bash
# 系统: 4 x NVIDIA H20
# 可用GPU: 0, 1, 2, 3
./incremental_experiment.sh --category single_object --new-prompts new.json --new-videos-dir videos/ --gpu-id 2
```

### 8卡系统示例  
```bash
# 系统: 8 x NVIDIA V100
# 可用GPU: 0, 1, 2, 3, 4, 5, 6, 7
./incremental_experiment.sh --category single_object --new-prompts new.json --new-videos-dir videos/ --gpu-id 7
```

### 自动选择示例
```bash
# 在任意系统上自动选择空闲GPU
./incremental_experiment.sh --category single_object --new-prompts new.json --new-videos-dir videos/

# 脚本会自动:
# 1. 检测系统GPU数量
# 2. 检查哪些GPU正在使用
# 3. 选择第一个空闲的GPU
```

## 🛡️ 错误处理

### GPU ID 验证
```bash
# 在4卡系统中指定GPU 7 (无效)
./incremental_experiment.sh --gpu-id 7 ...

# 输出:
# 错误: 指定的GPU ID (7) 超出系统GPU数量 (0-3)
```

### 自动降级
```bash
# 如果所有GPU都在使用中
# 脚本会显示警告并使用GPU 0
# 警告: 所有GPU都在使用中，将使用GPU 0
```

## 🔍 技术实现

### GPU数量检测
```bash
get_max_gpu_id() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo $((gpu_count - 1))
    else
        echo 0
    fi
}
```

### 动态范围循环
```bash
# 替换固定的 {0..3} 为动态循环
for ((gpu_id=0; gpu_id<=max_gpu_id; gpu_id++)); do
    # 检查GPU是否空闲
done
```

### 会话模式识别
```bash
# 支持识别多种会话命名模式
if [[ $session =~ gpu([0-9]+)_experiment ]] || [[ $session =~ incremental_.*_gpu([0-9]+) ]]; then
    busy_gpus+=(${BASH_REMATCH[1]})
fi
```

## 📈 优势

1. **兼容性**: 支持任意数量的GPU系统
2. **智能化**: 自动检测和验证GPU配置
3. **容错性**: 优雅处理无效的GPU ID
4. **向后兼容**: 原有的4卡使用方式完全不变
5. **扩展性**: 可以轻松支持未来的多GPU配置

## 🎯 使用建议

1. **首次使用**: 运行 `./test_gpu_detection.sh` 确认GPU检测正常
2. **自动模式**: 优先使用自动GPU选择，让脚本智能分配
3. **手动指定**: 只在需要特定GPU时手动指定GPU ID
4. **监控使用**: 通过tmux会话监控GPU使用情况
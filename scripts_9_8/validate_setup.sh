#!/bin/bash
# validate_setup.sh - 新主机配置验证脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_message $GREEN "🧪 开始环境验证..."

# 检查配置文件
if [ ! -f "config.env" ]; then
    print_message $RED "❌ config.env 文件不存在"
    exit 1
fi

# 加载配置
source config.env

# 1. 检查基础环境
print_message $BLUE "1️⃣ 检查基础环境..."
python3 --version || { print_message $RED "❌ Python3未安装"; exit 1; }
nvidia-smi > /dev/null || { print_message $RED "❌ NVIDIA驱动未安装"; exit 1; }
conda --version > /dev/null || { print_message $RED "❌ Conda未安装"; exit 1; }
print_message $GREEN "✅ 基础环境检查通过"

# 2. 检查目录结构
print_message $BLUE "2️⃣ 检查目录结构..."
required_dirs=(
    "$EXPERIMENT_BASE_DIR"
    "$EXPERIMENT_OUTPUT_DIR" 
    "$EXPERIMENT_LOG_DIR"
    "$MODEL_BASE_DIR"
    "$EXPERIMENT_DATA_DIR"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir"
    else
        print_message $RED "  ❌ $dir 不存在"
        exit 1
    fi
done

# 3. 检查关键脚本
print_message $BLUE "3️⃣ 检查关键脚本..."
required_files=(
    "FastVMT_incremental.py"
    "parallel_experiment_manager.sh"
    "incremental_experiment.sh"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        print_message $RED "  ❌ $file 不存在"
        exit 1
    fi
done

# 4. 检查conda环境 (如果存在)
print_message $BLUE "4️⃣ 检查conda环境..."
if conda env list | grep -q "$CONDA_ENV"; then
    print_message $GREEN "✅ Conda环境 '$CONDA_ENV' 存在"
    
    # 激活环境并检查包
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null && echo "  ✅ PyTorch已安装" || echo "  ⚠️  PyTorch未安装"
    python3 -c "import diffsynth; print('  DiffSynth导入成功')" 2>/dev/null && echo "  ✅ DiffSynth已安装" || echo "  ⚠️  DiffSynth未安装"
else
    print_message $YELLOW "⚠️  Conda环境 '$CONDA_ENV' 不存在，需要手动创建"
fi

# 5. 检查GPU
print_message $BLUE "5️⃣ 检查GPU..."
gpu_count=$(nvidia-smi --list-gpus | wc -l)
if [ $gpu_count -gt 0 ]; then
    print_message $GREEN "✅ 检测到 $gpu_count 个GPU"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  GPU $line"
    done
else
    print_message $RED "❌ 未检测到GPU"
fi

print_message $GREEN "🎉 基础环境验证完成！"
print_message $YELLOW "💡 下一步:"
print_message $YELLOW "  1. 如需要，创建conda环境: conda create -n $CONDA_ENV python=3.9"
print_message $YELLOW "  2. 复制数据集到: $EXPERIMENT_DATA_DIR"
print_message $YELLOW "  3. 复制模型到: $MODEL_BASE_DIR"
print_message $YELLOW "  4. 运行功能测试: ./test_parallel_setup.sh"

#!/bin/bash
# 新主机自动配置脚本

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

# 使用说明
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "🚀 新主机自动配置脚本"
    echo ""
    echo "用法: $0 <基础路径> [conda路径]"
    echo ""
    echo "参数:"
    echo "  基础路径    实验数据存储的基础目录"
    echo "  conda路径   conda安装路径 (可选，默认自动检测)"
    echo ""
    echo "示例:"
    echo "  $0 /data/experiments"
    echo "  $0 /home/user/projects /opt/miniconda3"
    echo ""
    exit 1
fi

BASE_PATH="$1"
CONDA_PATH="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_message $GREEN "🚀 开始配置新主机环境..."
print_message $BLUE "基础路径: $BASE_PATH"
print_message $BLUE "脚本目录: $SCRIPT_DIR"

# 检测conda路径
if [ -z "$CONDA_PATH" ]; then
    print_message $YELLOW "自动检测conda路径..."
    
    # 常见conda路径
    CONDA_CANDIDATES=(
        "/opt/miniconda3"
        "/root/miniconda3"
        "/home/$USER/miniconda3"
        "/usr/local/miniconda3"
        "$HOME/anaconda3"
        "/opt/anaconda3"
    )
    
    for path in "${CONDA_CANDIDATES[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/conda" ]; then
            CONDA_PATH="$path"
            print_message $GREEN "检测到conda: $CONDA_PATH"
            break
        fi
    done
    
    if [ -z "$CONDA_PATH" ]; then
        print_message $RED "未能自动检测到conda，请手动指定路径"
        exit 1
    fi
fi

# 创建目录结构
print_message $BLUE "1. 创建目录结构..."
mkdir -p "$BASE_PATH"/{logs,results_final_2,pretrained_models,Final_Dataset}
mkdir -p "$BASE_PATH"/logs/gpu_{0..7}

print_message $GREEN "✅ 目录创建完成"

# 复制配置模板
print_message $BLUE "2. 生成配置文件..."
if [ ! -f "$SCRIPT_DIR/config.env" ]; then
    cp "$SCRIPT_DIR/config.env.template" "$SCRIPT_DIR/config.env"
    print_message $GREEN "✅ 配置文件已创建"
else
    print_message $YELLOW "⚠️  config.env 已存在，将备份为 config.env.backup"
    cp "$SCRIPT_DIR/config.env" "$SCRIPT_DIR/config.env.backup"
    cp "$SCRIPT_DIR/config.env.template" "$SCRIPT_DIR/config.env"
fi

# 更新配置文件
print_message $BLUE "3. 更新配置文件..."

# 转义路径中的特殊字符
ESCAPED_BASE_PATH=$(echo "$BASE_PATH" | sed 's/[[\.*^$()+?{|]/\\&/g')
ESCAPED_SCRIPT_DIR=$(echo "$SCRIPT_DIR" | sed 's/[[\.*^$()+?{|]/\\&/g')
ESCAPED_CONDA_PATH=$(echo "$CONDA_PATH" | sed 's/[[\.*^$()+?{|]/\\&/g')

# 使用sed更新配置文件
sed -i "s|export EXPERIMENT_BASE_DIR=\".*\"|export EXPERIMENT_BASE_DIR=\"$BASE_PATH\"|" "$SCRIPT_DIR/config.env"
sed -i "s|export EXPERIMENT_WORK_DIR=\".*\"|export EXPERIMENT_WORK_DIR=\"$SCRIPT_DIR\"|" "$SCRIPT_DIR/config.env"
sed -i "s|export CONDA_BASE=\".*\"|export CONDA_BASE=\"$CONDA_PATH\"|" "$SCRIPT_DIR/config.env"

print_message $GREEN "✅ 配置文件更新完成"

# 设置脚本权限
print_message $BLUE "4. 设置脚本权限..."
chmod +x "$SCRIPT_DIR"/*.sh
print_message $GREEN "✅ 脚本权限设置完成"

# 显示配置信息
print_message $BLUE "5. 配置信息确认..."
echo ""
echo "📋 配置信息:"
echo "  基础目录: $BASE_PATH"
echo "  工作目录: $SCRIPT_DIR"
echo "  Conda路径: $CONDA_PATH"
echo "  输出目录: $BASE_PATH/results_final_2"
echo "  日志目录: $BASE_PATH/logs"
echo "  数据集目录: $BASE_PATH/Final_Dataset"
echo "  模型目录: $BASE_PATH/pretrained_models"
echo ""

# 创建验证脚本
print_message $BLUE "6. 创建验证脚本..."
cat > "$SCRIPT_DIR/validate_setup.sh" << 'EOF'
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
EOF

chmod +x "$SCRIPT_DIR/validate_setup.sh"
print_message $GREEN "✅ 验证脚本创建完成"

# 完成提示
print_message $GREEN "🎉 新主机配置完成！"
echo ""
print_message $YELLOW "📝 下一步操作:"
echo "1. 检查并编辑配置文件:"
echo "   vim $SCRIPT_DIR/config.env"
echo ""
echo "2. 运行环境验证:"
echo "   cd $SCRIPT_DIR"
echo "   ./validate_setup.sh"
echo ""
echo "3. 复制必要的数据文件:"
echo "   - 数据集 → $BASE_PATH/Final_Dataset/"
echo "   - 模型文件 → $BASE_PATH/pretrained_models/"
echo ""
echo "4. 创建并配置conda环境 (如果需要):"
echo "   conda create -n diffsynth python=3.9"
echo "   conda activate diffsynth"
echo "   # 安装必要的包..."
echo ""
echo "5. 运行功能测试:"
echo "   ./test_parallel_setup.sh"
echo ""
print_message $GREEN "🚀 配置完成，可以开始使用了！"
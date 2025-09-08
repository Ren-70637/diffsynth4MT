#!/bin/bash

# 灵活的GPU视频生成实验启动脚本
# 支持动态GPU选择、路径配置和增量实验

set -e  # 遇到错误立即退出

# 加载配置文件工具
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_config.sh"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# 默认参数
CONFIG_FILE="config.json"
SPECIFIED_GPUS=""
CATEGORIES=""
MODE="batch"
NEW_DATA_DIR=""
NEW_PROMPTS=""
ENABLE_MONITORING=true
DRY_RUN=false

# 检测可用GPU数量
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --list-gpus | wc -l
    else
        echo 0
    fi
}

# 显示GPU状态
show_gpu_status() {
    print_message $BLUE "当前GPU状态:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        while IFS=',' read -r index name util mem_used mem_total temp; do
            local mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "0")
            printf "${CYAN}GPU %s${NC}: %s | 利用率: %s%% | 显存: %s MB / %s MB (%.1f%%) | 温度: %s°C\n" \
                "$(echo $index | xargs)" "$(echo $name | xargs)" \
                "$(echo $util | tr -d ' %')" "$(echo $mem_used | xargs)" \
                "$(echo $mem_total | xargs)" "$mem_percent" "$(echo $temp | tr -d ' °C')"
        done
    else
        print_message $RED "nvidia-smi 不可用"
    fi
    echo ""
}

# 检查环境
check_environment() {
    print_message $BLUE "检查运行环境..."
    
    # 检查conda环境
    if ! command -v conda &> /dev/null; then
        print_message $RED "错误: conda未安装或未在PATH中"
        exit 1
    fi
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        print_message $RED "错误: nvidia-smi不可用"
        exit 1
    fi
    
    # 检查可用GPU数量
    local gpu_count=$(detect_gpus)
    if [ $gpu_count -eq 0 ]; then
        print_message $RED "错误: 没有检测到可用的GPU"
        exit 1
    fi
    
    print_message $GREEN "环境检查通过 - 发现 $gpu_count 个GPU"
}

# 检查必要文件
check_files() {
    print_message $BLUE "检查必要文件..."
    
    # 检查灵活脚本
    if [ ! -f "FastVMT_flexible.py" ]; then
        print_message $RED "错误: FastVMT_flexible.py 文件不存在"
        exit 1
    fi
    
    # 使用配置验证路径
    if ! validate_config; then
        print_message $RED "配置验证失败"
        exit 1
    fi
    
    print_message $GREEN "文件检查通过"
}

# 创建必要目录
create_experiment_directories() {
    print_message $BLUE "创建输出目录..."
    
    # 使用配置文件中的目录创建函数
    create_directories
    
    print_message $GREEN "目录创建完成"
}

# 检查tmux
check_tmux() {
    if ! command -v tmux &> /dev/null; then
        print_message $YELLOW "tmux未安装，正在安装..."
        sudo apt-get update && sudo apt-get install -y tmux
    fi
    print_message $GREEN "tmux可用"
}

# 启动灵活实验
start_flexible_experiment() {
    local session_name="flexible_experiment_$(date +%s)"
    
    print_message $BLUE "启动灵活实验"
    print_message $CYAN "模式: $MODE"
    print_message $CYAN "配置文件: $CONFIG_FILE"
    
    if [ -n "$SPECIFIED_GPUS" ]; then
        print_message $CYAN "指定GPU: $SPECIFIED_GPUS"
    else
        print_message $CYAN "GPU选择: 自动检测"
    fi
    
    if [ -n "$CATEGORIES" ]; then
        print_message $CYAN "指定类别: $CATEGORIES"
    else
        print_message $CYAN "类别: 使用配置文件"
    fi
    
    # 构建Python命令
    local python_cmd="python FastVMT_flexible.py --config $CONFIG_FILE --mode $MODE"
    
    if [ -n "$SPECIFIED_GPUS" ]; then
        python_cmd="$python_cmd --gpus $SPECIFIED_GPUS"
    fi
    
    if [ -n "$CATEGORIES" ]; then
        python_cmd="$python_cmd --categories $CATEGORIES"
    fi
    
    if [ -n "$NEW_DATA_DIR" ]; then
        python_cmd="$python_cmd --new-data-dir $NEW_DATA_DIR"
    fi
    
    if [ -n "$NEW_PROMPTS" ]; then
        python_cmd="$python_cmd --new-prompts $NEW_PROMPTS"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_message $YELLOW "干运行模式 - 将要执行的命令:"
        echo "$python_cmd"
        return
    fi
    
    # 杀死已存在的会话
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # 创建新的tmux会话
    tmux new-session -d -s "$session_name"
    
    # 在会话中执行命令
    tmux send-keys -t "$session_name" "source $(get_config conda_path)" C-m
    tmux send-keys -t "$session_name" "conda activate $(get_config conda_env)" C-m
    tmux send-keys -t "$session_name" "cd $(get_config work_dir)" C-m
    
    print_message $PURPLE "执行命令: $python_cmd"
    tmux send-keys -t "$session_name" "$python_cmd" C-m
    
    print_message $GREEN "灵活实验已在tmux会话 '$session_name' 中启动"
    
    # 启动监控窗口（如果启用）
    if [ "$ENABLE_MONITORING" = true ]; then
        tmux new-window -t "$session_name" -n "monitor"
        tmux send-keys -t "$session_name:monitor" "watch -n 5 nvidia-smi" C-m
        print_message $CYAN "监控窗口已启动"
    fi
    
    # 显示管理命令
    print_message $YELLOW "管理命令:"
    echo -e "${YELLOW}连接到实验会话: tmux attach-session -t $session_name${NC}"
    echo -e "${YELLOW}查看所有会话: tmux list-sessions${NC}"
    echo -e "${YELLOW}停止实验: tmux kill-session -t $session_name${NC}"
    echo -e "${YELLOW}在会话内切换窗口: Ctrl+B 然后按 1,2,3...${NC}"
    echo -e "${YELLOW}退出会话(保持运行): Ctrl+B 然后按 D${NC}"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --gpus)
                SPECIFIED_GPUS="$2"
                shift 2
                ;;
            --categories)
                CATEGORIES="$2"
                shift 2
                ;;
            --mode)
                MODE="$2"
                if [[ ! "$MODE" =~ ^(batch|incremental|single)$ ]]; then
                    print_message $RED "错误: 无效的模式 '$MODE'，必须是 batch、incremental 或 single"
                    exit 1
                fi
                shift 2
                ;;
            --new-data-dir)
                NEW_DATA_DIR="$2"
                shift 2
                ;;
            --new-prompts)
                NEW_PROMPTS="$2"
                shift 2
                ;;
            --no-monitor)
                ENABLE_MONITORING=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --status)
                show_gpu_status
                exit 0
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_message $RED "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 显示帮助信息
show_help() {
    cat << EOF
灵活的GPU视频生成实验启动脚本

用法: $0 [OPTIONS]

选项:
  --config FILE               配置文件路径 (默认: config.json)
  --gpus "0 1 2"             手动指定使用的GPU ID列表
  --categories "cat1 cat2"    指定处理的类别
  --mode MODE                 运行模式: batch|incremental|single (默认: batch)
  --new-data-dir DIR          新增数据目录 (增量模式)
  --new-prompts FILE          新增prompts文件 (增量模式)
  --no-monitor               不启动监控窗口
  --dry-run                  干运行模式，只显示命令不执行
  --status                   显示GPU状态并退出
  --help, -h                 显示此帮助信息

运行模式说明:
  batch       批量处理所有数据 (默认)
  incremental 增量处理，跳过已存在的结果
  single      单GPU模式

示例:
  $0                                    # 使用默认配置自动选择GPU
  $0 --gpus "0 1"                      # 指定使用GPU 0和1
  $0 --mode incremental                # 增量模式
  $0 --categories "camera_motion"      # 只处理摄像机运动类别
  $0 --dry-run                         # 预览将要执行的命令
  $0 --status                          # 查看GPU状态

GPU选择规则:
  - 如果指定了 --gpus，使用指定的GPU
  - 否则自动检测可用GPU并选择满足显存要求的GPU
  - 排除配置文件中exclude_gpus列表的GPU
  - 优先选择显存较大且空闲的GPU

路径配置:
  - 所有路径支持相对路径（相对于工作目录）和绝对路径
  - 相对路径会自动转换为基于配置文件中work_dir的绝对路径
  - 便于实验环境迁移
EOF
}

# 验证参数
validate_args() {
    # 检查增量模式的必需参数
    if [ "$MODE" = "incremental" ]; then
        if [ -z "$NEW_DATA_DIR" ] && [ -z "$NEW_PROMPTS" ]; then
            print_message $YELLOW "警告: 增量模式建议指定 --new-data-dir 或 --new-prompts"
        fi
        
        if [ -n "$NEW_DATA_DIR" ] && [ ! -d "$NEW_DATA_DIR" ]; then
            print_message $RED "错误: 新数据目录不存在: $NEW_DATA_DIR"
            exit 1
        fi
        
        if [ -n "$NEW_PROMPTS" ] && [ ! -f "$NEW_PROMPTS" ]; then
            print_message $RED "错误: 新prompts文件不存在: $NEW_PROMPTS"
            exit 1
        fi
    fi
    
    # 验证配置文件
    if [ ! -f "$CONFIG_FILE" ]; then
        print_message $RED "错误: 配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
}

# 显示实验配置摘要
show_experiment_summary() {
    print_message $BLUE "实验配置摘要"
    echo "========================================"
    echo "配置文件: $CONFIG_FILE"
    echo "运行模式: $MODE"
    echo "GPU选择: $([ -n "$SPECIFIED_GPUS" ] && echo "手动指定: $SPECIFIED_GPUS" || echo "自动检测")"
    echo "类别过滤: $([ -n "$CATEGORIES" ] && echo "$CATEGORIES" || echo "使用配置文件")"
    echo "监控窗口: $([ "$ENABLE_MONITORING" = true ] && echo "启用" || echo "禁用")"
    
    if [ "$MODE" = "incremental" ]; then
        echo "新数据目录: $([ -n "$NEW_DATA_DIR" ] && echo "$NEW_DATA_DIR" || echo "未指定")"
        echo "新prompts: $([ -n "$NEW_PROMPTS" ] && echo "$NEW_PROMPTS" || echo "未指定")"
    fi
    
    echo "========================================"
    echo ""
}

# 主函数
main() {
    # 解析命令行参数
    parse_args "$@"
    
    # 验证参数
    validate_args
    
    print_message $GREEN "启动灵活的GPU视频生成实验系统"
    print_message $BLUE "="*60
    
    # 显示实验配置摘要
    show_experiment_summary
    
    # 加载配置文件
    if ! load_config "$CONFIG_FILE"; then
        print_message $RED "配置文件加载失败"
        exit 1
    fi
    
    # 环境检查
    check_environment
    check_files
    check_tmux
    create_experiment_directories
    
    # 切换到工作目录
    cd "$(get_config work_dir)"
    
    # 显示GPU状态
    show_gpu_status
    
    # 启动实验
    start_flexible_experiment
    
    print_message $GREEN "灵活实验系统启动完成!"
    print_message $BLUE "="*60
}

# 执行主函数
main "$@"

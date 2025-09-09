#!/bin/bash

# 增量实验脚本 - 专门处理新增数据
# 用法: ./incremental_experiment.sh [category] [new_prompts_file] [new_videos_dir]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# 默认参数
CATEGORY=""
NEW_PROMPTS_FILE=""
NEW_VIDEOS_DIR=""
OUTPUT_DIR="${EXPERIMENT_OUTPUT_DIR:-$(pwd)/results_final_2}"
BASE_PROMPTS_FILE="${EXPERIMENT_PROMPTS_FILE:-$(pwd)/Final_Dataset/prompt.json}"
VIDEO_RANGE=""        # 新增: 视频范围，如"1-7"或"8-14"
INSTANCE_ID=""        # 新增: 实例ID，用于区分同类别的多个实验

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --new-prompts)
            NEW_PROMPTS_FILE="$2"
            shift 2
            ;;
        --new-videos-dir)
            NEW_VIDEOS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --video-range)
            VIDEO_RANGE="$2"
            shift 2
            ;;
        --instance-id)
            INSTANCE_ID="$2"
            shift 2
            ;;
        --help|-h)
            echo "增量实验脚本用法:"
            echo "$0 --category <类别> --new-prompts <新prompts文件> --new-videos-dir <新视频目录>"
            echo ""
            echo "参数:"
            echo "  --category        指定类别 (如: single_object)"
            echo "  --new-prompts     新增的prompts JSON文件路径"
            echo "  --new-videos-dir  新增视频所在目录"
            echo "  --output-dir      输出目录 (默认: \$EXPERIMENT_OUTPUT_DIR 或 $(pwd)/results_final_2)"
            echo "  --gpu-id          指定使用的GPU ID (默认: 自动选择空闲GPU)"
            echo "  --video-range     视频范围 (如: 1-7, 8-14, 可选)"
            echo "  --instance-id     实例ID (用于区分同类别多实验, 可选)"
            echo ""
            echo "环境变量配置:"
            echo "  EXPERIMENT_BASE_DIR      实验基础目录"
            echo "  EXPERIMENT_OUTPUT_DIR    输出目录"
            echo "  EXPERIMENT_PROMPTS_FILE  主prompts文件路径"
            echo "  MODEL_BASE_DIR           模型文件目录"
            echo "  CONDA_BASE               Conda安装路径"
            echo "  CONDA_ENV                Conda环境名"
            echo ""
            echo "使用方法:"
            echo "1. 使用配置文件: source config.env && $0 --category single_object ..."
            echo "2. 临时设置: EXPERIMENT_BASE_DIR=/path $0 --category single_object ..."
            echo "3. 直接指定: $0 --category single_object --output-dir /custom/path ..."
            echo ""
            echo "示例:"
            echo "$0 --category single_object --new-prompts /path/to/new_prompts.json --new-videos-dir /path/to/new/videos"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看用法"
            exit 1
            ;;
    esac
done

# 参数验证
if [[ -z "$CATEGORY" ]]; then
    print_message $RED "错误: 必须指定 --category 参数"
    exit 1
fi

if [[ -z "$NEW_PROMPTS_FILE" ]]; then
    print_message $RED "错误: 必须指定 --new-prompts 参数"
    exit 1
fi

if [[ -z "$NEW_VIDEOS_DIR" ]]; then
    print_message $RED "错误: 必须指定 --new-videos-dir 参数"
    exit 1
fi

# 检查文件和目录
if [[ ! -f "$NEW_PROMPTS_FILE" ]]; then
    print_message $RED "错误: 新prompts文件不存在 - $NEW_PROMPTS_FILE"
    exit 1
fi

if [[ ! -d "$NEW_VIDEOS_DIR" ]]; then
    print_message $RED "错误: 新视频目录不存在 - $NEW_VIDEOS_DIR"
    exit 1
fi

# 自动选择GPU
select_gpu() {
    if [[ -n "$GPU_ID" ]]; then
        # 验证指定的GPU ID是否有效
        local max_gpu_id=$(get_max_gpu_id)
        if [[ $GPU_ID -gt $max_gpu_id ]]; then
            print_message $RED "错误: 指定的GPU ID ($GPU_ID) 超出系统GPU数量 (0-$max_gpu_id)"
            exit 1
        fi
        echo $GPU_ID
        return
    fi
    
    print_message $BLUE "自动选择空闲GPU..."
    
    # 获取系统GPU数量
    local max_gpu_id=$(get_max_gpu_id)
    print_message $BLUE "检测到GPU数量: $((max_gpu_id + 1))"
    
    # 检查哪些GPU正在运行实验
    local busy_gpus=()
    for session in $(tmux list-sessions -F "#{session_name}" 2>/dev/null || echo ""); do
        if [[ $session =~ gpu([0-9]+)_experiment ]] || [[ $session =~ incremental_.*_gpu([0-9]+) ]]; then
            busy_gpus+=(${BASH_REMATCH[1]})
        fi
    done
    
    # 找到第一个空闲的GPU
    for ((gpu_id=0; gpu_id<=max_gpu_id; gpu_id++)); do
        local is_busy=false
        for busy_gpu in "${busy_gpus[@]}"; do
            if [[ $gpu_id -eq $busy_gpu ]]; then
                is_busy=true
                break
            fi
        done
        
        if [[ $is_busy == false ]]; then
            echo $gpu_id
            return
        fi
    done
    
    print_message $YELLOW "警告: 所有GPU都在使用中，将使用GPU 0"
    echo 0
}

# 获取系统最大GPU ID
get_max_gpu_id() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo $((gpu_count - 1))
    else
        echo 0
    fi
}

# 合并prompts文件
'''merge_prompts() {
    local temp_prompts="/tmp/merged_prompts_$$.json"
    
    # 静默执行，不显示颜色
    echo "合并prompts文件..." >&2
    
    # 使用Python脚本合并JSON文件并支持视频范围过滤
    python3 << EOF
import json

# 读取原始prompts
with open('$BASE_PROMPTS_FILE', 'r', encoding='utf-8') as f:
    base_prompts = json.load(f)

# 读取新增prompts
with open('$NEW_PROMPTS_FILE', 'r', encoding='utf-8') as f:
    new_prompts = json.load(f)

# 合并到指定类别
if '$CATEGORY' not in base_prompts:
    base_prompts['$CATEGORY'] = {}

if '$CATEGORY' in new_prompts:
    base_prompts['$CATEGORY'].update(new_prompts['$CATEGORY'])

# 视频范围过滤
video_range = '$VIDEO_RANGE'
if video_range and '$CATEGORY' in base_prompts:
    try:
        if '-' in video_range:
            start, end = map(int, video_range.split('-'))
            video_keys = list(base_prompts['$CATEGORY'].keys())
            # 转换为0索引
            selected_keys = video_keys[start-1:end]
            filtered_prompts = {k: base_prompts['$CATEGORY'][k] for k in selected_keys if k in base_prompts['$CATEGORY']}
            base_prompts['$CATEGORY'] = filtered_prompts
            pass  # 静默处理视频范围过滤
        else:
            pass  # 静默处理格式错误
    except Exception as e:
        pass  # 静默处理解析错误

# 保存合并后的文件
with open('$temp_prompts', 'w', encoding='utf-8') as f:
    json.dump(base_prompts, f, ensure_ascii=False, indent=2)
EOF
    
    echo $temp_prompts
}
'''
# 启动增量实验
start_incremental_experiment() {
    local gpu_id=$1
    local merged_prompts=$2
    local session_suffix=""
    if [[ -n "$INSTANCE_ID" ]]; then
        session_suffix="_${INSTANCE_ID}"
    fi
    local session_name="incremental_${CATEGORY}${session_suffix}_gpu${gpu_id}"
    
    # 清理路径，移除颜色码
    merged_prompts=$(echo "$merged_prompts" | sed 's/\x1b\[[0-9;]*m//g')
    
    print_message $BLUE "启动增量实验: GPU $gpu_id - $CATEGORY"
    
    # 杀死已存在的会话
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # 创建新的tmux会话
    tmux new-session -d -s "$session_name"
    
    # 在会话中执行命令
    local conda_base="${CONDA_BASE:-/root/miniconda3}"
    local conda_env="${CONDA_ENV:-diffsynth}"
    local work_dir="${EXPERIMENT_WORK_DIR:-$(pwd)}"

    tmux send-keys -t \"$session_name\" \"source ${work_dir}/config.env\" C-m
    tmux send-keys -t "$session_name" "source ${conda_base}/etc/profile.d/conda.sh" C-m
    tmux send-keys -t "$session_name" "conda activate ${conda_env}" C-m
    tmux send-keys -t "$session_name" "cd ${work_dir}" C-m
    
    # 构建Python命令 - 只处理新增数据
    local python_cmd="python FastVMT_incremental.py --gpu_id $gpu_id --category '$CATEGORY' --output_dir '$OUTPUT_DIR' --ref_prompts_path '$merged_prompts' --base_path '$NEW_VIDEOS_DIR' --model_base_dir \"$MODEL_BASE_DIR\" --log_base_dir \"$EXPERIMENT_LOG_DIR\" --num_persistent 50000000 --incremental_mode"
    
    tmux send-keys -t "$session_name" "$python_cmd" C-m
    
    print_message $GREEN "增量实验已在tmux会话 '$session_name' 中启动"
    print_message $YELLOW "查看进度: tmux attach-session -t $session_name"
}

# 主函数
main() {
    print_message $GREEN "开始增量实验准备"
    print_message $BLUE "="*60
    
    print_message $BLUE "参数信息:"
    print_message $BLUE "类别: $CATEGORY"
    print_message $BLUE "新prompts文件: $NEW_PROMPTS_FILE"
    print_message $BLUE "新视频目录: $NEW_VIDEOS_DIR"
    print_message $BLUE "输出目录: $OUTPUT_DIR"
    [[ -n "$VIDEO_RANGE" ]] && print_message $BLUE "视频范围: $VIDEO_RANGE"
    [[ -n "$INSTANCE_ID" ]] && print_message $BLUE "实例ID: $INSTANCE_ID"
    
    # 选择GPU
    selected_gpu=$(select_gpu)
    print_message $GREEN "选择GPU: $selected_gpu"
    
    # 合并prompts
    # merged_prompts=$(merge_prompts)
    merged_prompts=$NEW_PROMPTS_FILE
    print_message $GREEN "Prompts合并完成: $merged_prompts"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 启动实验
    start_incremental_experiment $selected_gpu $merged_prompts
    
    print_message $GREEN "增量实验启动完成!"
    print_message $BLUE "="*60
    
    # 清理临时文件
    # rm -f $merged_prompts  # 暂时保留用于调试
    
    print_message $YELLOW "管理命令:"
    echo -e "${YELLOW}查看会话: tmux attach-session -t incremental_${CATEGORY}_gpu${selected_gpu}${NC}"
    echo -e "${YELLOW}查看所有会话: tmux list-sessions${NC}"
    echo -e "${YELLOW}停止会话: tmux kill-session -t incremental_${CATEGORY}_gpu${selected_gpu}${NC}"
}

# 执行主函数
main "$@"

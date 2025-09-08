#!/bin/bash

# 并行实验管理器
# 自动分配多GPU并行视频生成实验

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
MODE=""
DATASET_PATH=""
PROMPTS_FILE=""
GPU_COUNT=0
PROCESSES_PER_GPU=1  # 新增: 每GPU运行的实验进程数

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --prompts)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --processes-per-gpu)
            PROCESSES_PER_GPU="$2"
            shift 2
            ;;
        --help|-h)
            echo "并行实验管理器用法:"
            echo "$0 --mode <模式> --dataset <数据集路径>"
            echo ""
            echo "模式选项:"
            echo "  4gpu              4卡服务器模式 (每GPU一个类别)"
            echo "  8gpu-conservative 8卡服务器保守模式 (每GPU一个类别)"
            echo "  8gpu-aggressive   8卡服务器激进模式 (部分GPU多任务)"
            echo "  custom            自定义模式"
            echo ""
            echo "参数:"
            echo "  --dataset             数据集基础路径 (默认: Final_Dataset)"
            echo "  --prompts             prompts文件路径 (默认: dataset/prompt.json)"
            echo "  --processes-per-gpu   每GPU运行的实验进程数 (默认: 1)"
            echo ""
            echo "示例:"
            echo "$0 --mode 4gpu --dataset Final_Dataset"
            echo "$0 --mode 4gpu --dataset Final_Dataset --processes-per-gpu 2"
            echo "$0 --mode 8gpu-aggressive --dataset Final_Dataset --processes-per-gpu 3"
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
if [[ -z "$MODE" ]]; then
    print_message $RED "错误: 必须指定 --mode 参数"
    exit 1
fi

# 默认值设置
DATASET_PATH="${DATASET_PATH:-Final_Dataset}"
PROMPTS_FILE="${PROMPTS_FILE:-${DATASET_PATH}/prompt.json}"

# 检测GPU数量
detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_message $GREEN "检测到 $GPU_COUNT 个GPU"
    else
        print_message $RED "错误: 未检测到NVIDIA GPU"
        exit 1
    fi
}

# 分析数据集结构
analyze_dataset() {
    print_message $BLUE "分析数据集结构..."
    
    if [[ ! -f "$PROMPTS_FILE" ]]; then
        print_message $RED "错误: prompts文件不存在 - $PROMPTS_FILE"
        exit 1
    fi
    
    # 使用Python分析数据集
    python3 << EOF
import json
import os

try:
    with open('$PROMPTS_FILE', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    print("数据集分析结果:")
    total_videos = 0
    total_tasks = 0
    
    for category, videos in prompts.items():
        video_count = len(videos)
        # 每个视频5个prompt，每个prompt 2个seed
        tasks = video_count * 5 * 2
        total_videos += video_count
        total_tasks += tasks
        print(f"  {category}: {video_count} 个视频, {tasks} 个任务")
    
    print(f"总计: {total_videos} 个视频, {total_tasks} 个任务")
    
    # 输出分类信息供shell使用
    categories = list(prompts.keys())
    print("CATEGORIES=" + ",".join(categories))
    
except Exception as e:
    print(f"分析失败: {e}")
    exit(1)
EOF
}

# 启动单个实验进程的辅助函数
start_single_experiment() {
    local gpu_id=$1
    local category=$2
    local instance_id=$3
    local video_range=$4
    
    print_message $YELLOW "启动GPU $gpu_id: $category (实例: $instance_id)"
    
    local cmd="./incremental_experiment.sh \
        --category '$category' \
        --new-prompts '$PROMPTS_FILE' \
        --new-videos-dir '${DATASET_PATH}/${category}' \
        --gpu-id $gpu_id \
        --instance-id '$instance_id'"
    
    if [[ -n "$video_range" ]]; then
        cmd+=" --video-range '$video_range'"
    fi
    
    eval "$cmd" &
    sleep 2  # 避免同时启动冲突
}

# 4GPU模式任务分配
setup_4gpu_mode() {
    print_message $BLUE "配置4GPU模式 (每GPU ${PROCESSES_PER_GPU} 个进程)..."
    
    local categories=("camera_motion" "single_object" "multiple_objects" "complex_human_motion")
    
    if [[ $PROCESSES_PER_GPU -eq 1 ]]; then
        # 传统模式：每GPU一个类别
        for i in {0..3}; do
            start_single_experiment $i "${categories[$i]}" "gpu${i}" ""
        done
    else
        # 多进程模式：每个类别按视频范围拆分
        for i in {0..3}; do
            local category=${categories[$i]}
            local gpu_id=$i
            
            # 使用Python计算视频拆分
            python3 << EOF | while IFS=':' read -r range_id video_range; do
import json

try:
    with open('$PROMPTS_FILE', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    category = "$category"
    processes_per_gpu = $PROCESSES_PER_GPU
    
    if category in prompts:
        videos = list(prompts[category].keys())
        total_videos = len(videos)
        videos_per_process = max(1, total_videos // processes_per_gpu)
        
        for p in range(processes_per_gpu):
            start_idx = p * videos_per_process + 1
            if p == processes_per_gpu - 1:  # 最后一个进程处理剩余所有视频
                end_idx = total_videos
            else:
                end_idx = min((p + 1) * videos_per_process, total_videos)
            
            if start_idx <= total_videos:
                print(f"{p+1}:{start_idx}-{end_idx}")
    else:
        print("1:1-1")  # 默认范围
except Exception as e:
    print("1:1-1")  # 错误时使用默认范围
EOF
                start_single_experiment $gpu_id "$category" "proc${range_id}" "$video_range"
            done
        done
    fi
    
    wait
    print_message $GREEN "4GPU并行实验启动完成"
}

# 8GPU保守模式 
setup_8gpu_conservative() {
    print_message $BLUE "配置8GPU保守模式 (每GPU ${PROCESSES_PER_GPU} 个进程)..."
    
    local categories=("camera_motion" "single_object" "multiple_objects" "complex_human_motion")
    
    if [[ $PROCESSES_PER_GPU -eq 1 ]]; then
        # 传统模式：前4个GPU处理完整类别，后4个GPU处理副本
        for i in {0..3}; do
            start_single_experiment $i "${categories[$i]}" "full" ""
        done
        
        for i in {4..7}; do
            local category_idx=$((i - 4))
            start_single_experiment $i "${categories[$category_idx]}" "backup" ""
        done
    else
        # 多进程模式：所有8个GPU均匀分布任务
        for gpu_id in {0..7}; do
            local category_idx=$((gpu_id % 4))
            local category=${categories[$category_idx]}
            
            # 每个类别在2个GPU上运行，每GPU运行multiple进程
            python3 << EOF | while IFS=':' read -r range_id video_range; do
import json

try:
    with open('$PROMPTS_FILE', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    category = "$category"
    gpu_id = $gpu_id
    processes_per_gpu = $PROCESSES_PER_GPU
    
    if category in prompts:
        videos = list(prompts[category].keys())
        total_videos = len(videos)
        
        # 8GPU模式：每个类别分配给2个GPU，每GPU处理一半数据
        gpu_group = gpu_id // 4  # 0 或 1
        videos_per_gpu_group = total_videos // 2
        
        if gpu_group == 0:
            gpu_start = 1
            gpu_end = videos_per_gpu_group
        else:
            gpu_start = videos_per_gpu_group + 1
            gpu_end = total_videos
        
        # 在GPU组内再按进程数拆分
        gpu_videos = gpu_end - gpu_start + 1
        videos_per_process = max(1, gpu_videos // processes_per_gpu)
        
        for p in range(processes_per_gpu):
            start_idx = gpu_start + p * videos_per_process
            if p == processes_per_gpu - 1:
                end_idx = gpu_end
            else:
                end_idx = min(gpu_start + (p + 1) * videos_per_process - 1, gpu_end)
            
            if start_idx <= total_videos:
                print(f"{p+1}:{start_idx}-{end_idx}")
    else:
        print("1:1-1")
except Exception as e:
    print("1:1-1")
EOF
                start_single_experiment $gpu_id "$category" "g${gpu_id}p${range_id}" "$video_range"
            done
        done
    fi
    
    wait
    print_message $GREEN "8GPU保守模式启动完成"
}

# 8GPU激进模式
setup_8gpu_aggressive() {
    print_message $BLUE "配置8GPU激进模式..."
    
    local categories=("camera_motion" "single_object" "multiple_objects" "complex_human_motion")
    
    # 获取每个类别的视频数量并分割
    python3 << EOF > /tmp/video_splits.txt
import json

with open('$PROMPTS_FILE', 'r', encoding='utf-8') as f:
    prompts = json.load(f)

categories = ["camera_motion", "single_object", "multiple_objects", "complex_human_motion"]

for i, category in enumerate(categories):
    if category in prompts:
        videos = list(prompts[category].keys())
        video_count = len(videos)
        mid_point = video_count // 2
        
        # GPU i: 前半部分
        print(f"GPU{i}:{category}:1-{mid_point}")
        # GPU i+4: 后半部分  
        print(f"GPU{i+4}:{category}:{mid_point+1}-{video_count}")
    else:
        print(f"GPU{i}:{category}:1-1")
        print(f"GPU{i+4}:{category}:1-1")
EOF

    # 读取分割方案并启动任务
    while IFS=':' read -r gpu_assignment category video_range; do
        local gpu_id=${gpu_assignment#GPU}
        
        print_message $YELLOW "启动$gpu_assignment: $category (视频范围: $video_range)"
        
        ./incremental_experiment.sh \
            --category "$category" \
            --new-prompts "$PROMPTS_FILE" \
            --new-videos-dir "${DATASET_PATH}/${category}" \
            --gpu-id $gpu_id \
            --video-range "$video_range" \
            --instance-id "split${video_range}" &
        
        sleep 2
        
    done < /tmp/video_splits.txt
    
    rm -f /tmp/video_splits.txt
    wait
    print_message $GREEN "8GPU激进模式启动完成"
}

# 显示运行状态
show_status() {
    print_message $BLUE "当前实验状态:"
    
    echo "活跃的tmux会话:"
    tmux list-sessions 2>/dev/null | grep -E "(incremental_|gpu.*_experiment)" || echo "  无活跃实验会话"
    
    echo ""
    echo "GPU使用情况:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=',' read -r gpu_id name mem_used mem_total util; do
        printf "  GPU %d (%s): %s/%s MB 显存, %s%% 利用率\n" \
               "$gpu_id" "$name" "$mem_used" "$mem_total" "$util"
    done
}

# 主函数
main() {
    print_message $GREEN "启动并行实验管理器"
    print_message $BLUE "="*60
    
    # 检测硬件
    detect_gpu_count
    
    # 显示配置信息
    print_message $BLUE "配置信息:"
    print_message $BLUE "  模式: $MODE"
    print_message $BLUE "  GPU数量: $GPU_COUNT"
    print_message $BLUE "  每GPU进程数: $PROCESSES_PER_GPU"
    print_message $BLUE "  总并行数: $((GPU_COUNT * PROCESSES_PER_GPU))"
    print_message $BLUE "  数据集路径: $DATASET_PATH"
    
    # 分析数据集
    analyze_dataset
    
    # 验证模式与GPU数量匹配
    case $MODE in
        "4gpu")
            if [[ $GPU_COUNT -lt 4 ]]; then
                print_message $RED "错误: 4GPU模式需要至少4个GPU，当前只有 $GPU_COUNT 个"
                exit 1
            fi
            setup_4gpu_mode
            ;;
        "8gpu-conservative"|"8gpu-aggressive")
            if [[ $GPU_COUNT -lt 8 ]]; then
                print_message $RED "错误: 8GPU模式需要至少8个GPU，当前只有 $GPU_COUNT 个"
                exit 1
            fi
            if [[ $MODE == "8gpu-conservative" ]]; then
                setup_8gpu_conservative
            else
                setup_8gpu_aggressive
            fi
            ;;
        "custom")
            print_message $YELLOW "自定义模式: 请手动配置实验"
            echo "可用的incremental_experiment.sh参数示例:"
            echo "./incremental_experiment.sh --category camera_motion --video-range 1-7 --gpu-id 0 --instance-id part1"
            ;;
        *)
            print_message $RED "错误: 未知模式 '$MODE'"
            exit 1
            ;;
    esac
    
    # 显示状态
    print_message $BLUE "="*60
    show_status
    
    print_message $GREEN "并行实验管理器完成!"
    print_message $YELLOW "使用 'tmux list-sessions' 查看实验会话"
    print_message $YELLOW "使用 'tmux attach-session -t <session_name>' 连接到具体实验"
}

# 执行主函数
main "$@"
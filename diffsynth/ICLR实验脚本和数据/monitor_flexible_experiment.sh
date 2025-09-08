#!/bin/bash

# 灵活实验监控脚本 - 支持各种实验模式的监控

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# 检测可用GPU数量
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --list-gpus | wc -l
    else
        echo 0
    fi
}

# 显示GPU状态概览
show_gpu_overview() {
    print_message $BLUE "GPU状态概览"
    echo "========================================"
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_message $RED "nvidia-smi 不可用"
        return
    fi
    
    # 显示GPU基本信息
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
    while IFS=',' read -r index name util mem_used mem_total temp power; do
        # 计算内存使用率
        local mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "0")
        
        printf "${CYAN}GPU %s${NC}: %s\n" "$index" "$(echo $name | xargs)"
        printf "  利用率: %s%% | 显存: %s MB / %s MB (%.1f%%) | 温度: %s°C | 功耗: %s W\n" \
            "$(echo $util | tr -d ' %')" \
            "$(echo $mem_used | xargs)" \
            "$(echo $mem_total | xargs)" \
            "$mem_percent" \
            "$(echo $temp | tr -d ' °C')" \
            "$(echo $power | tr -d ' W')"
        echo ""
    done
}

# 显示tmux会话状态
show_session_status() {
    print_message $BLUE "Tmux会话状态"
    echo "========================================"
    
    local sessions=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -E "flexible_experiment|gpu[0-9]+_(parallel|monitor)" | sort || true)
    
    if [ -z "$sessions" ]; then
        print_message $YELLOW "没有找到运行中的实验会话"
        return
    fi
    
    echo "$sessions" | while read session; do
        if [ -n "$session" ]; then
            local windows=$(tmux list-windows -t "$session" -F "#{window_name}" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
            local session_info=$(tmux display-message -t "$session" -p "#{session_created}")
            
            if [[ "$session" == flexible_experiment* ]]; then
                printf "${GREEN}灵活实验会话${NC}: %s | 窗口: %s | 创建时间: %s\n" "$session" "$windows" "$session_info"
            elif [[ "$session" == *"_parallel" ]]; then
                printf "${PURPLE}并行会话${NC}: %s | 窗口: %s\n" "$session" "$windows"
            else
                printf "${CYAN}监控会话${NC}: %s | 窗口: %s\n" "$session" "$windows"
            fi
        fi
    done
    echo ""
}

# 显示进程状态
show_process_status() {
    print_message $BLUE "进程状态"
    echo "========================================"
    
    # 查找Python进程
    local python_processes=$(ps aux | grep -E "(FastVMT_flexible|FastVMT_multi_gpu_parallel|FastVMT_incremental)" | grep -v grep || true)
    
    if [ -n "$python_processes" ]; then
        echo "运行中的实验进程:"
        printf "${CYAN}%-8s %-8s %-6s %-6s %-10s %s${NC}\n" "USER" "PID" "%CPU" "%MEM" "TIME" "COMMAND"
        echo "$python_processes" | while read line; do
            local user=$(echo "$line" | awk '{print $1}')
            local pid=$(echo "$line" | awk '{print $2}')
            local cpu=$(echo "$line" | awk '{print $3}')
            local mem=$(echo "$line" | awk '{print $4}')
            local time=$(echo "$line" | awk '{print $10}')
            local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}' | cut -c1-50)
            
            printf "%-8s %-8s %-6s %-6s %-10s %s\n" "$user" "$pid" "$cpu" "$mem" "$time" "$cmd"
        done
    else
        print_message $YELLOW "没有发现运行中的实验进程"
    fi
    echo ""
}

# 显示实验进度（通用版本）
show_experiment_progress() {
    local config_file=${1:-"config.json"}
    
    print_message $BLUE "实验进度"
    echo "========================================"
    
    if [ ! -f "$config_file" ]; then
        print_message $RED "配置文件不存在: $config_file"
        return
    fi
    
    # 检查是否为增量配置
    local is_incremental=$(jq -r '.incremental_data // empty' "$config_file" 2>/dev/null)
    
    if [ -n "$is_incremental" ]; then
        # 增量实验进度
        local category=$(jq -r '.incremental_data.category' "$config_file")
        local timestamp=$(jq -r '.incremental_data.timestamp' "$config_file")
        
        printf "${PURPLE}增量实验${NC}: %s\n" "$category"
        printf "创建时间: %s\n" "$timestamp"
        echo ""
    fi
    
    # 检查输出目录
    local output_dir=$(jq -r '.paths.output_dir' "$config_file" 2>/dev/null)
    if [ "$output_dir" = "null" ]; then
        print_message $YELLOW "无法从配置文件读取输出目录"
        return
    fi
    
    # 如果是相对路径，转换为绝对路径
    if [[ ! "$output_dir" = /* ]]; then
        local work_dir=$(jq -r '.environment.work_dir' "$config_file" 2>/dev/null)
        if [ "$work_dir" != "null" ]; then
            output_dir="$work_dir/$output_dir"
        fi
    fi
    
    if [ ! -d "$output_dir" ]; then
        print_message $YELLOW "输出目录不存在: $output_dir"
        return
    fi
    
    # 统计各类别的进度
    for category_dir in "$output_dir"/*; do
        if [ -d "$category_dir" ]; then
            local category=$(basename "$category_dir")
            printf "${CYAN}类别 %s${NC}:\n" "$category"
            
            # 统计GPU和进程的输出
            local total_videos=0
            for gpu_process_dir in "$category_dir"/gpu_*_process_*; do
                if [ -d "$gpu_process_dir" ]; then
                    local gpu_process=$(basename "$gpu_process_dir")
                    local video_count=$(find "$gpu_process_dir" -name "*.mp4" 2>/dev/null | wc -l)
                    total_videos=$((total_videos + video_count))
                    printf "  %s: %d 个视频\n" "$gpu_process" "$video_count"
                fi
            done
            
            # 检查非GPU分组的输出
            local other_videos=$(find "$category_dir" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)
            if [ $other_videos -gt 0 ]; then
                total_videos=$((total_videos + other_videos))
                printf "  其他: %d 个视频\n" "$other_videos"
            fi
            
            printf "  ${GREEN}总计: %d 个视频${NC}\n" "$total_videos"
            echo ""
        fi
    done
}

# 显示最新日志
show_recent_logs() {
    local lines=${1:-10}
    local log_pattern=${2:-"logs"}
    
    print_message $BLUE "最新日志 (最后 $lines 行)"
    echo "========================================"
    
    # 查找最新的日志文件
    local log_files=$(find $log_pattern -name "*.log" 2>/dev/null | head -5)
    
    if [ -z "$log_files" ]; then
        print_message $YELLOW "没有找到日志文件"
        return
    fi
    
    echo "$log_files" | while read log_file; do
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            printf "${GREEN}%s${NC}:\n" "$(basename "$log_file")"
            tail -n "$lines" "$log_file" | sed 's/^/  /'
            echo ""
        fi
    done
}

# 实时监控模式
real_time_monitor() {
    local interval=${1:-5}
    local config_file=${2:-"config.json"}
    
    print_message $GREEN "启动实时监控模式 (刷新间隔: ${interval}秒)"
    print_message $YELLOW "按 Ctrl+C 退出监控"
    echo ""
    
    while true; do
        clear
        echo "================ 灵活实验实时监控 ================"
        echo "刷新时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        show_gpu_overview
        show_session_status
        show_process_status
        show_experiment_progress "$config_file"
        
        echo "======================================================="
        echo "按 Ctrl+C 退出监控 | 下次刷新: ${interval}秒后"
        
        sleep "$interval"
    done
}

# 生成监控报告
generate_report() {
    local config_file=${1:-"config.json"}
    local report_file="flexible_experiment_report_$(date '+%Y%m%d_%H%M%S').txt"
    
    print_message $BLUE "生成监控报告: $report_file"
    
    {
        echo "================ 灵活实验监控报告 ================"
        echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        echo "系统信息:"
        echo "主机名: $(hostname)"
        echo "GPU数量: $(detect_gpus)"
        echo ""
        
        echo "GPU状态概览:"
        nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
        echo ""
        
        echo "会话状态:"
        tmux list-sessions 2>/dev/null || echo "Tmux未运行"
        echo ""
        
        echo "进程状态:"
        ps aux | grep -E "(FastVMT_flexible|FastVMT_multi_gpu_parallel|FastVMT_incremental)" | grep -v grep || echo "没有运行的实验进程"
        echo ""
        
        echo "实验进度:"
        if [ -f "$config_file" ]; then
            # 这里需要一个纯文本版本的进度显示
            local output_dir=$(jq -r '.paths.output_dir' "$config_file" 2>/dev/null)
            if [ "$output_dir" != "null" ] && [ -d "$output_dir" ]; then
                find "$output_dir" -name "*.mp4" | wc -l | xargs echo "总视频数量:"
            fi
        fi
        echo ""
        
        echo "======================================================="
    } > "$report_file"
    
    print_message $GREEN "报告已保存到: $report_file"
}

# 智能检测实验类型
detect_experiment_type() {
    local sessions=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | head -1)
    
    if [[ "$sessions" == flexible_experiment* ]]; then
        echo "flexible"
    elif [[ "$sessions" == *"parallel"* ]]; then
        echo "parallel"
    elif [[ "$sessions" == incremental* ]]; then
        echo "incremental"
    else
        echo "unknown"
    fi
}

# 显示帮助信息
show_help() {
    echo "灵活实验监控脚本"
    echo ""
    echo "用法: $0 [OPTIONS]"
    echo ""
    echo "选项:"
    echo "  --overview             显示GPU状态概览"
    echo "  --sessions             显示tmux会话状态"
    echo "  --processes            显示进程状态"
    echo "  --progress [CONFIG]    显示实验进度"
    echo "  --logs [LINES]         显示最新日志 (默认10行)"
    echo "  --monitor [INTERVAL]   实时监控模式 (默认5秒刷新)"
    echo "  --report [CONFIG]      生成监控报告"
    echo "  --config FILE          指定配置文件 (默认: config.json)"
    echo "  --auto                 自动检测实验类型并显示相应信息"
    echo "  --help, -h             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                     # 显示完整状态"
    echo "  $0 --auto              # 自动检测并显示相关信息"
    echo "  $0 --monitor 3         # 3秒刷新的实时监控"
    echo "  $0 --logs 20           # 显示最新20行日志"
    echo "  $0 --report            # 生成报告"
    echo "  $0 --config config_incremental_*.json  # 使用增量配置"
}

# 自动模式 - 智能显示相关信息
auto_mode() {
    local experiment_type=$(detect_experiment_type)
    
    print_message $BLUE "自动检测到实验类型: $experiment_type"
    echo ""
    
    case $experiment_type in
        flexible)
            print_message $GREEN "检测到灵活实验会话"
            # 尝试找到相关的配置文件
            local config_files=$(ls config_incremental_*.json 2>/dev/null | head -1)
            if [ -n "$config_files" ]; then
                show_experiment_progress "$config_files"
            else
                show_experiment_progress "config.json"
            fi
            ;;
        parallel)
            print_message $GREEN "检测到并行实验会话"
            show_experiment_progress "config.json"
            ;;
        incremental)
            print_message $GREEN "检测到增量实验会话"
            local config_files=$(ls config_incremental_*.json 2>/dev/null | head -1)
            if [ -n "$config_files" ]; then
                show_experiment_progress "$config_files"
            fi
            ;;
        *)
            print_message $YELLOW "未检测到活跃的实验会话"
            show_experiment_progress "config.json"
            ;;
    esac
    
    echo ""
    show_session_status
    show_process_status
}

# 主函数
main() {
    local config_file="config.json"
    
    case "${1:-}" in
        --overview)
            show_gpu_overview
            ;;
        --sessions)
            show_session_status
            ;;
        --processes)
            show_process_status
            ;;
        --progress)
            local target_config="${2:-$config_file}"
            show_experiment_progress "$target_config"
            ;;
        --logs)
            local lines=${2:-10}
            show_recent_logs "$lines"
            ;;
        --monitor)
            local interval=${2:-5}
            local target_config=${3:-$config_file}
            real_time_monitor "$interval" "$target_config"
            ;;
        --report)
            local target_config="${2:-$config_file}"
            generate_report "$target_config"
            ;;
        --config)
            config_file="${2:-config.json}"
            show_experiment_progress "$config_file"
            ;;
        --auto)
            auto_mode
            ;;
        --help|-h)
            show_help
            ;;
        "")
            # 默认显示完整状态
            show_gpu_overview
            show_session_status
            show_process_status
            show_experiment_progress "$config_file"
            ;;
        *)
            print_message $RED "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"

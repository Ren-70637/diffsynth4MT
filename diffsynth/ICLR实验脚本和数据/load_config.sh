#!/bin/bash

# 配置文件加载工具脚本
# 提供bash脚本读取JSON配置文件的功能

# 默认配置文件路径
DEFAULT_CONFIG_FILE="config.json"

# 全局变量存储配置
declare -A CONFIG

# 加载配置文件
load_config() {
    local config_file="${1:-$DEFAULT_CONFIG_FILE}"
    
    if [ ! -f "$config_file" ]; then
        echo "错误: 配置文件不存在 - $config_file"
        return 1
    fi
    
    # 检查jq是否可用
    if ! command -v jq &> /dev/null; then
        echo "错误: 需要安装jq来解析JSON配置文件"
        echo "安装命令: sudo apt-get install jq"
        return 1
    fi
    
    echo "加载配置文件: $config_file"
    
    # 读取环境配置
    CONFIG[conda_path]=$(jq -r '.environment.conda_path' "$config_file")
    CONFIG[conda_env]=$(jq -r '.environment.conda_env' "$config_file")
    CONFIG[work_dir]=$(jq -r '.environment.work_dir' "$config_file")
    
    # 读取路径配置
    CONFIG[base_path]=$(jq -r '.paths.base_path' "$config_file")
    CONFIG[prompt_file]=$(jq -r '.paths.prompt_file' "$config_file")
    CONFIG[output_dir]=$(jq -r '.paths.output_dir' "$config_file")
    CONFIG[log_dir]=$(jq -r '.paths.log_dir' "$config_file")
    
    # 读取类别路径
    CONFIG[camera_motion_path]=$(jq -r '.paths.categories.camera_motion' "$config_file")
    CONFIG[single_object_path]=$(jq -r '.paths.categories.single_object' "$config_file")
    CONFIG[multiple_objects_path]=$(jq -r '.paths.categories.multiple_objects' "$config_file")
    CONFIG[complex_human_motion_path]=$(jq -r '.paths.categories.complex_human_motion' "$config_file")
    
    # 读取模型配置
    CONFIG[model_base_path]=$(jq -r '.model.base_path' "$config_file")
    
     # 读取实验配置
     CONFIG[gpu_ids]=$(jq -r 'if .experiment.gpu_ids then .experiment.gpu_ids | join(",") else "" end' "$config_file")
     CONFIG[num_persistent]=$(jq -r '.experiment.generation_params.num_persistent' "$config_file")
    
    # 输出加载的关键配置
    echo "配置加载完成:"
    echo "  工作目录: ${CONFIG[work_dir]}"
    echo "  conda环境: ${CONFIG[conda_env]}"
    echo "  数据集路径: ${CONFIG[base_path]}"
    echo "  输出目录: ${CONFIG[output_dir]}"
    echo "  模型路径: ${CONFIG[model_base_path]}"
    
    return 0
}

# 获取配置值
get_config() {
    local key="$1"
    echo "${CONFIG[$key]}"
}

# 获取GPU对应的类别
get_gpu_category() {
    local gpu_id="$1"
    local config_file="${2:-$DEFAULT_CONFIG_FILE}"
    
    jq -r ".experiment.gpu_categories.\"$gpu_id\"" "$config_file"
}

# 验证关键路径
validate_config() {
    local all_valid=true
    
    echo "验证配置路径..."
    
    # 检查关键路径
    local paths=(
        "base_path:数据集基础路径"
        "prompt_file:prompt文件"
        "model_base_path:模型基础路径"
    )
    
    for path_info in "${paths[@]}"; do
        local key="${path_info%%:*}"
        local desc="${path_info##*:}"
        local path="${CONFIG[$key]}"
        
        if [ ! -e "$path" ]; then
            echo "错误: $desc 不存在 - $path"
            all_valid=false
        else
            echo "✓ $desc: $path"
        fi
    done
    
    # 检查类别目录
    local categories=("camera_motion" "single_object" "multiple_objects" "complex_human_motion")
    for category in "${categories[@]}"; do
        local path="${CONFIG[${category}_path]}"
        if [ ! -d "$path" ]; then
            echo "警告: ${category}目录不存在 - $path"
        else
            echo "✓ ${category}目录: $path"
        fi
    done
    
    if [ "$all_valid" = true ]; then
        echo "配置验证通过!"
        return 0
    else
        echo "配置验证失败!"
        return 1
    fi
}

# 创建必要目录
create_directories() {
    local dirs=(
        "${CONFIG[output_dir]}"
        "${CONFIG[log_dir]}"
        "${CONFIG[log_dir]}/gpu_0"
        "${CONFIG[log_dir]}/gpu_1"
        "${CONFIG[log_dir]}/gpu_2"
        "${CONFIG[log_dir]}/gpu_3"
    )
    
    echo "创建必要目录..."
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "创建目录: $dir"
        else
            echo "目录已存在: $dir"
        fi
    done
}

# 应用命令行参数覆盖
apply_args_override() {
    # 这个函数可以被其他脚本调用来应用命令行参数覆盖配置
    while [[ $# -gt 0 ]]; do
        case $1 in
            --base-path)
                CONFIG[base_path]="$2"
                shift 2
                ;;
            --output-dir)
                CONFIG[output_dir]="$2"
                shift 2
                ;;
            --model-path)
                CONFIG[model_base_path]="$2"
                shift 2
                ;;
            --conda-env)
                CONFIG[conda_env]="$2"
                shift 2
                ;;
            --prompt-file)
                CONFIG[prompt_file]="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
}

# 显示帮助信息
show_config_help() {
    echo "配置文件工具脚本"
    echo ""
    echo "用法:"
    echo "  source load_config.sh"
    echo "  load_config [config_file]"
    echo ""
    echo "函数:"
    echo "  load_config [file]     - 加载配置文件 (默认: config.json)"
    echo "  get_config key         - 获取配置值"
    echo "  get_gpu_category id    - 获取GPU对应的类别"
    echo "  validate_config        - 验证配置路径"
    echo "  create_directories     - 创建必要目录"
    echo "  apply_args_override    - 应用命令行参数覆盖"
    echo ""
    echo "支持的命令行参数覆盖:"
    echo "  --base-path PATH       - 数据集基础路径"
    echo "  --output-dir PATH      - 输出目录"
    echo "  --model-path PATH      - 模型基础路径"
    echo "  --conda-env ENV        - conda环境名"
    echo "  --prompt-file FILE     - prompt文件路径"
}

# 如果直接运行脚本，则显示帮助
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    show_config_help
fi
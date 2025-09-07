#!/usr/bin/env python3
"""
实验进度检查工具
用于查看当前实验的进度，识别已完成和未完成的任务
"""

import os
import json
import argparse
from collections import defaultdict

def count_results_in_directory(directory):
    """统计目录中的结果文件数量"""
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                count += 1
    return count

def analyze_category_progress(output_dir, category, prompts_data):
    """分析特定类别的进度"""
    category_dir = os.path.join(output_dir, category)
    
    if category not in prompts_data:
        print(f"❌ 类别 {category} 在prompts中不存在")
        return
    
    videos = prompts_data[category]
    total_expected = 0
    completed = 0
    
    print(f"\n📊 类别: {category}")
    print("-" * 60)
    
    for video_key, prompts in videos.items():
        video_dir = os.path.join(category_dir, video_key)
        
        for i, prompt in enumerate(prompts):
            prompt_dir = os.path.join(video_dir, f"prompt{i+1}")
            
            # 每个prompt预期生成2个视频（2个seed）
            expected_files = 2
            total_expected += expected_files
            
            actual_files = count_results_in_directory(prompt_dir)
            completed += min(actual_files, expected_files)
            
            status = "✅" if actual_files >= expected_files else f"⚠️ ({actual_files}/{expected_files})"
            print(f"{status} {video_key} - prompt{i+1}: {actual_files}/{expected_files}")
    
    completion_rate = (completed / total_expected * 100) if total_expected > 0 else 0
    print(f"\n📈 进度: {completed}/{total_expected} ({completion_rate:.1f}%)")
    
    return {
        'category': category,
        'completed': completed,
        'total': total_expected,
        'rate': completion_rate
    }

def find_missing_tasks(output_dir, category, prompts_data):
    """查找缺失的任务"""
    if category not in prompts_data:
        return []
    
    missing_tasks = []
    category_dir = os.path.join(output_dir, category)
    videos = prompts_data[category]
    
    for video_key, prompts in videos.items():
        for i, prompt in enumerate(prompts):
            prompt_dir = os.path.join(category_dir, f"prompt{i+1}")
            actual_files = count_results_in_directory(prompt_dir)
            
            if actual_files < 2:  # 期望每个prompt有2个视频
                missing_tasks.append({
                    'video_key': video_key,
                    'prompt_index': i + 1,
                    'completed_files': actual_files,
                    'missing_files': 2 - actual_files
                })
    
    return missing_tasks

def generate_resume_script(missing_tasks, category, output_dir, base_path, prompts_path):
    """生成续传脚本"""
    if not missing_tasks:
        return None
    
    script_path = f"/root/autodl-tmp/diffsynth4MT/resume_{category}_experiment.sh"
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# 续传 {category} 类别的实验\n")
        f.write("# 自动生成的脚本\n\n")
        f.write("set -e\n\n")
        
        f.write("# 激活环境\n")
        f.write("source /root/miniconda3/etc/profile.d/conda.sh\n")
        f.write("conda activate diffsynth\n")
        f.write("cd /root/autodl-tmp/diffsynth4MT\n\n")
        
        f.write("# 选择空闲GPU\n")
        f.write("select_gpu() {\n")
        f.write("    # 自动检测GPU数量\n")
        f.write("    local max_gpu_id=0\n")
        f.write("    if command -v nvidia-smi &> /dev/null; then\n")
        f.write("        local gpu_count=$(nvidia-smi --list-gpus | wc -l)\n")
        f.write("        max_gpu_id=$((gpu_count - 1))\n")
        f.write("    fi\n")
        f.write("    \n")
        f.write("    for ((gpu_id=0; gpu_id<=max_gpu_id; gpu_id++)); do\n")
        f.write("        if ! tmux list-sessions 2>/dev/null | grep -q \"gpu${gpu_id}_experiment\\|incremental_.*_gpu${gpu_id}\"; then\n")
        f.write("            echo $gpu_id\n")
        f.write("            return\n")
        f.write("        fi\n")
        f.write("    done\n")
        f.write("    echo 0  # 默认使用GPU 0\n")
        f.write("}\n\n")
        
        f.write("GPU_ID=$(select_gpu)\n")
        f.write("echo \"使用GPU: $GPU_ID\"\n\n")
        
        f.write("# 启动续传实验\n")
        f.write(f"python FastVMT_incremental.py \\\n")
        f.write(f"    --gpu_id $GPU_ID \\\n")
        f.write(f"    --category {category} \\\n")
        f.write(f"    --output_dir {output_dir} \\\n")
        f.write(f"    --ref_prompts_path {prompts_path} \\\n")
        f.write(f"    --base_path {base_path} \\\n")
        f.write(f"    --num_persistent 50000000 \\\n")
        f.write(f"    --incremental_mode\n")
    
    os.chmod(script_path, 0o755)
    return script_path

def main():
    parser = argparse.ArgumentParser(description="检查实验进度")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/results_final_2", 
                      help="实验输出目录")
    parser.add_argument("--prompts_path", type=str, default="/root/autodl-tmp/Final_Dataset/prompt.json",
                      help="prompts文件路径")
    parser.add_argument("--category", type=str, help="指定要检查的类别")
    parser.add_argument("--generate_resume", action="store_true", help="生成续传脚本")
    parser.add_argument("--base_path", type=str, default="/root/autodl-tmp/Final_Dataset",
                      help="视频文件基础路径")
    
    args = parser.parse_args()
    
    # 读取prompts
    try:
        with open(args.prompts_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取prompts文件失败: {e}")
        return
    
    print("🔍 实验进度检查报告")
    print("=" * 80)
    
    if args.category:
        # 检查指定类别
        categories = [args.category]
    else:
        # 检查所有类别
        categories = list(prompts_data.keys())
    
    total_stats = {'completed': 0, 'total': 0}
    
    for category in categories:
        stats = analyze_category_progress(args.output_dir, category, prompts_data)
        if stats:
            total_stats['completed'] += stats['completed']
            total_stats['total'] += stats['total']
        
        # 查找缺失任务并生成续传脚本
        if args.generate_resume:
            missing_tasks = find_missing_tasks(args.output_dir, category, prompts_data)
            if missing_tasks:
                category_base_path = os.path.join(args.base_path, category)
                script_path = generate_resume_script(
                    missing_tasks, category, args.output_dir, 
                    category_base_path, args.prompts_path
                )
                print(f"\n📝 生成续传脚本: {script_path}")
                print(f"   缺失任务数: {len(missing_tasks)}")
                print(f"   运行命令: ./{os.path.basename(script_path)}")
            else:
                print(f"\n✅ {category} 类别所有任务已完成")
    
    # 总体统计
    if len(categories) > 1:
        overall_rate = (total_stats['completed'] / total_stats['total'] * 100) if total_stats['total'] > 0 else 0
        print(f"\n🎯 总体进度: {total_stats['completed']}/{total_stats['total']} ({overall_rate:.1f}%)")
    
    print("\n" + "=" * 80)
    print("检查完成!")

if __name__ == "__main__":
    main()
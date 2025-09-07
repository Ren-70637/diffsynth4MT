#!/usr/bin/env python3
"""
增量实验管理器 - 支持动态添加新实验数据
"""

import json
import os
import shutil
import argparse
import time
from datetime import datetime

def backup_existing_data(source_dir, backup_dir):
    """备份现有数据"""
    if not os.path.exists(source_dir):
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
    
    print(f"备份现有数据到: {backup_path}")
    shutil.copytree(source_dir, backup_path, dirs_exist_ok=True)
    
    return backup_path

def merge_prompts(base_prompts_file, new_prompts_file, output_file=None):
    """合并prompts文件"""
    # 读取基础prompts
    if os.path.exists(base_prompts_file):
        with open(base_prompts_file, 'r', encoding='utf-8') as f:
            base_prompts = json.load(f)
    else:
        base_prompts = []
    
    # 读取新增prompts
    with open(new_prompts_file, 'r', encoding='utf-8') as f:
        new_prompts = json.load(f)
    
    # 创建视频名称到prompt的映射
    base_mapping = {item['video_name']: item for item in base_prompts}
    
    # 合并新prompts
    merged_prompts = list(base_prompts)
    added_count = 0
    updated_count = 0
    
    for new_item in new_prompts:
        video_name = new_item['video_name']
        if video_name in base_mapping:
            # 更新现有prompt
            base_mapping[video_name].update(new_item)
            updated_count += 1
        else:
            # 添加新prompt
            merged_prompts.append(new_item)
            added_count += 1
    
    # 保存合并后的文件
    if output_file is None:
        output_file = base_prompts_file
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_prompts, f, ensure_ascii=False, indent=2)
    
    print(f"Prompts合并完成: 新增 {added_count} 个, 更新 {updated_count} 个")
    return output_file

def copy_new_videos(new_videos_dir, target_category_dir):
    """复制新视频文件到目标类别目录"""
    if not os.path.exists(new_videos_dir):
        print(f"新视频目录不存在: {new_videos_dir}")
        return 0
    
    os.makedirs(target_category_dir, exist_ok=True)
    
    copied_count = 0
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    
    for filename in os.listdir(new_videos_dir):
        if filename.lower().endswith(video_extensions):
            source_path = os.path.join(new_videos_dir, filename)
            target_path = os.path.join(target_category_dir, filename)
            
            if not os.path.exists(target_path):
                shutil.copy2(source_path, target_path)
                print(f"复制新视频: {filename}")
                copied_count += 1
            else:
                print(f"跳过已存在的视频: {filename}")
    
    print(f"复制完成: {copied_count} 个新视频文件")
    return copied_count

def scan_existing_results(results_dir):
    """扫描已存在的实验结果"""
    if not os.path.exists(results_dir):
        return set()
    
    existing_results = set()
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.mp4'):
                # 提取结果标识符
                relative_path = os.path.relpath(os.path.join(root, file), results_dir)
                existing_results.add(relative_path)
    
    return existing_results

def generate_incremental_config(base_config_file, incremental_data):
    """生成增量实验配置"""
    with open(base_config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新配置以支持增量模式
    config['experiment']['modes']['incremental']['enabled'] = True
    config['experiment']['modes']['incremental']['skip_existing'] = True
    
    # 添加增量数据信息
    config['incremental_data'] = incremental_data
    
    # 生成临时配置文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    incremental_config_file = f"config_incremental_{timestamp}.json"
    
    with open(incremental_config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return incremental_config_file

def prepare_incremental_experiment(args):
    """准备增量实验"""
    print("="*60)
    print("准备增量实验")
    print("="*60)
    
    # 读取基础配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    work_dir = config['environment']['work_dir']
    
    # 解析路径
    def resolve_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(work_dir, path)
    
    base_prompts_file = resolve_path(config['paths']['prompt_file'])
    backup_dir = resolve_path(config['paths'].get('backup_dir', './backup'))
    
    # 确定目标类别目录
    if args.category in config['paths']['categories']:
        target_category_dir = resolve_path(config['paths']['categories'][args.category])
    else:
        print(f"错误: 类别 '{args.category}' 在配置中不存在")
        return None
    
    incremental_data = {
        'category': args.category,
        'timestamp': datetime.now().isoformat(),
        'new_videos_dir': args.new_videos_dir,
        'new_prompts_file': args.new_prompts_file
    }
    
    # 步骤1: 备份现有数据（如果启用）
    if args.backup:
        backup_path = backup_existing_data(target_category_dir, backup_dir)
        incremental_data['backup_path'] = backup_path
    
    # 步骤2: 复制新视频文件
    if args.new_videos_dir:
        copied_count = copy_new_videos(args.new_videos_dir, target_category_dir)
        incremental_data['copied_videos'] = copied_count
    
    # 步骤3: 合并prompts文件
    if args.new_prompts_file:
        merged_prompts_file = merge_prompts(base_prompts_file, args.new_prompts_file)
        incremental_data['merged_prompts_file'] = merged_prompts_file
    
    # 步骤4: 扫描已存在的结果
    results_dir = resolve_path(config['paths']['output_dir'])
    existing_results = scan_existing_results(results_dir)
    incremental_data['existing_results_count'] = len(existing_results)
    
    print(f"检测到 {len(existing_results)} 个已存在的实验结果")
    
    # 步骤5: 生成增量配置文件
    incremental_config_file = generate_incremental_config(args.config, incremental_data)
    
    print("="*60)
    print("增量实验准备完成")
    print(f"增量配置文件: {incremental_config_file}")
    print(f"目标类别: {args.category}")
    if args.new_videos_dir:
        print(f"新视频数量: {incremental_data.get('copied_videos', 0)}")
    if args.new_prompts_file:
        print(f"Prompts文件: 已合并")
    print("="*60)
    
    return incremental_config_file

def show_incremental_status(config_file):
    """显示增量实验状态"""
    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if 'incremental_data' not in config:
        print("这不是增量实验配置文件")
        return
    
    incremental_data = config['incremental_data']
    
    print("="*60)
    print("增量实验状态")
    print("="*60)
    print(f"类别: {incremental_data['category']}")
    print(f"创建时间: {incremental_data['timestamp']}")
    print(f"新视频目录: {incremental_data.get('new_videos_dir', '未指定')}")
    print(f"新Prompts文件: {incremental_data.get('new_prompts_file', '未指定')}")
    print(f"复制的视频数量: {incremental_data.get('copied_videos', 0)}")
    print(f"已存在结果数量: {incremental_data.get('existing_results_count', 0)}")
    
    if 'backup_path' in incremental_data:
        print(f"备份路径: {incremental_data['backup_path']}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='增量实验管理器')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 准备命令
    prepare_parser = subparsers.add_parser('prepare', help='准备增量实验')
    prepare_parser.add_argument('--config', type=str, default='config.json', help='基础配置文件')
    prepare_parser.add_argument('--category', type=str, required=True, help='目标类别')
    prepare_parser.add_argument('--new-videos-dir', type=str, help='新视频目录')
    prepare_parser.add_argument('--new-prompts-file', type=str, help='新prompts文件')
    prepare_parser.add_argument('--backup', action='store_true', help='备份现有数据')
    
    # 状态命令
    status_parser = subparsers.add_parser('status', help='显示增量实验状态')
    status_parser.add_argument('config_file', help='增量配置文件')
    
    # 合并命令
    merge_parser = subparsers.add_parser('merge', help='合并prompts文件')
    merge_parser.add_argument('--base', type=str, required=True, help='基础prompts文件')
    merge_parser.add_argument('--new', type=str, required=True, help='新prompts文件')
    merge_parser.add_argument('--output', type=str, help='输出文件（默认覆盖基础文件）')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        incremental_config = prepare_incremental_experiment(args)
        if incremental_config:
            print(f"\n可以使用以下命令启动增量实验:")
            print(f"./start_flexible_experiment.sh --config {incremental_config} --mode incremental")
    
    elif args.command == 'status':
        show_incremental_status(args.config_file)
    
    elif args.command == 'merge':
        merge_prompts(args.base, args.new, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

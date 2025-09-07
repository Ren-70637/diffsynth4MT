#!/usr/bin/env python3
"""
实验状态检查和分析脚本
提供详细的实验进度、错误分析和性能统计
"""

import os
import json
import time
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import argparse

from experiment_utils import ExperimentManager

class ExperimentChecker:
    """实验检查器"""
    
    def __init__(self):
        self.manager = ExperimentManager()
        self.results_dir = Path("/root/autodl-tmp/experiment_results")
        self.logs_dir = self.results_dir / "logs"
        
    def get_gpu_status(self):
        """获取GPU状态"""
        try:
            cmd = [
                "nvidia-smi", 
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpu_info.append({
                                'id': int(parts[0]),
                                'name': parts[1],
                                'memory_used': int(parts[2]),
                                'memory_total': int(parts[3]),
                                'utilization': int(parts[4]),
                                'temperature': int(parts[5])
                            })
                return gpu_info
            else:
                return []
        except Exception as e:
            print(f"获取GPU状态失败: {e}")
            return []
    
    def get_tmux_sessions(self):
        """获取tmux会话状态"""
        try:
            result = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sessions = [s.strip() for s in result.stdout.split('\n') 
                           if s.strip().startswith('wan_')]
                return sessions
            else:
                return []
        except Exception:
            return []
    
    def analyze_progress(self):
        """分析实验进度"""
        progress_data = {}
        total_generated = 0
        total_expected = 0
        
        for gpu_id in range(4):
            category = self.manager.gpu_categories[gpu_id]
            category_dir = self.results_dir / category
            
            # 统计已生成视频
            if category_dir.exists():
                generated_files = list(category_dir.glob("**/*.mp4"))
                generated = len(generated_files)
            else:
                generated = 0
                generated_files = []
            
            # 计算预期数量
            video_files = self.manager.get_video_files_for_category(category)
            expected = 0
            video_details = {}
            
            for video_name in video_files:
                prompts = self.manager.get_prompts_for_video(video_name)
                video_expected = len(prompts) * len(self.manager.seeds)
                expected += video_expected
                
                # 统计该视频的完成情况
                video_dir = category_dir / video_name if category_dir.exists() else None
                if video_dir and video_dir.exists():
                    video_generated = len(list(video_dir.glob("*.mp4")))
                else:
                    video_generated = 0
                
                video_details[video_name] = {
                    'expected': video_expected,
                    'generated': video_generated,
                    'progress': video_generated / video_expected * 100 if video_expected > 0 else 0
                }
            
            progress_data[category] = {
                'gpu_id': gpu_id,
                'generated': generated,
                'expected': expected,
                'progress': generated / expected * 100 if expected > 0 else 0,
                'videos': video_details,
                'files': generated_files
            }
            
            total_generated += generated
            total_expected += expected
        
        return {
            'categories': progress_data,
            'total_generated': total_generated,
            'total_expected': total_expected,
            'total_progress': total_generated / total_expected * 100 if total_expected > 0 else 0
        }
    
    def analyze_logs(self):
        """分析日志文件"""
        log_analysis = {}
        
        if not self.logs_dir.exists():
            return log_analysis
        
        for log_file in self.logs_dir.glob("gpu*_process*.log"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 提取基本信息
                total_lines = len(lines)
                error_lines = [line for line in lines if 'ERROR' in line]
                warning_lines = [line for line in lines if 'WARNING' in line]
                completed_tasks = [line for line in lines if '任务完成:' in line]
                failed_tasks = [line for line in lines if '任务处理失败:' in line]
                
                # 最后活动时间
                last_activity = None
                if lines:
                    last_line = lines[-1].strip()
                    try:
                        # 尝试提取时间戳
                        if ' - ' in last_line:
                            timestamp_str = last_line.split(' - ')[0]
                            last_activity = timestamp_str
                    except:
                        pass
                
                log_analysis[log_file.name] = {
                    'total_lines': total_lines,
                    'errors': len(error_lines),
                    'warnings': len(warning_lines),
                    'completed_tasks': len(completed_tasks),
                    'failed_tasks': len(failed_tasks),
                    'last_activity': last_activity,
                    'recent_errors': error_lines[-3:] if error_lines else []
                }
                
            except Exception as e:
                log_analysis[log_file.name] = {'error': str(e)}
        
        return log_analysis
    
    def estimate_completion_time(self, progress_data):
        """估算完成时间"""
        total_generated = progress_data['total_generated']
        total_expected = progress_data['total_expected']
        
        if total_generated == 0:
            return "无法估算（尚无完成任务）"
        
        # 通过日志分析平均处理时间
        avg_time_per_task = 180  # 默认3分钟每任务
        
        try:
            # 尝试从日志中获取更准确的时间
            sample_times = []
            for log_file in self.logs_dir.glob("gpu*_process*.log"):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if '任务完成:' in line and '耗时:' in line:
                        try:
                            time_part = line.split('耗时:')[1].split('秒')[0].strip()
                            sample_times.append(float(time_part))
                        except:
                            continue
            
            if sample_times:
                avg_time_per_task = sum(sample_times) / len(sample_times)
                
        except Exception:
            pass
        
        remaining_tasks = total_expected - total_generated
        estimated_seconds = remaining_tasks * avg_time_per_task / 8  # 8个并行进程
        
        if estimated_seconds <= 0:
            return "即将完成"
        
        # 转换为小时和分钟
        hours = int(estimated_seconds // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        
        if hours > 0:
            return f"约 {hours} 小时 {minutes} 分钟"
        else:
            return f"约 {minutes} 分钟"
    
    def print_detailed_report(self):
        """打印详细报告"""
        print("=" * 80)
        print("Wan2.1-T2V-14B 实验详细状态报告")
        print("=" * 80)
        print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # GPU状态
        print("📊 GPU状态:")
        gpu_status = self.get_gpu_status()
        if gpu_status:
            for gpu in gpu_status:
                memory_percent = gpu['memory_used'] / gpu['memory_total'] * 100
                print(f"  GPU {gpu['id']}: {gpu['utilization']:3d}% | "
                      f"内存 {gpu['memory_used']:5d}/{gpu['memory_total']}MB ({memory_percent:.1f}%) | "
                      f"温度 {gpu['temperature']}°C")
        else:
            print("  无法获取GPU状态")
        print()
        
        # tmux会话状态
        print("🖥️  tmux会话状态:")
        sessions = self.get_tmux_sessions()
        if sessions:
            for session in sessions:
                print(f"  ✓ {session}")
        else:
            print("  ❌ 没有活动的实验会话")
        print()
        
        # 进度分析
        print("📈 实验进度:")
        progress = self.analyze_progress()
        print(f"  总体进度: {progress['total_generated']}/{progress['total_expected']} "
              f"({progress['total_progress']:.1f}%)")
        print()
        
        # 各类别详情
        for category, data in progress['categories'].items():
            print(f"  📁 {category} (GPU {data['gpu_id']}):")
            print(f"     进度: {data['generated']}/{data['expected']} ({data['progress']:.1f}%)")
            
            # 显示完成度最低的几个视频
            incomplete_videos = [(name, info) for name, info in data['videos'].items() 
                               if info['progress'] < 100]
            incomplete_videos.sort(key=lambda x: x[1]['progress'])
            
            if incomplete_videos:
                print(f"     未完成视频 (前5个):")
                for video_name, info in incomplete_videos[:5]:
                    print(f"       - {video_name}: {info['generated']}/{info['expected']} "
                          f"({info['progress']:.1f}%)")
            print()
        
        # 完成时间估算
        print("⏰ 预计完成时间:")
        completion_estimate = self.estimate_completion_time(progress)
        print(f"  {completion_estimate}")
        print()
        
        # 日志分析
        print("📋 日志分析:")
        log_analysis = self.analyze_logs()
        if log_analysis:
            total_errors = sum(log['errors'] for log in log_analysis.values() if 'errors' in log)
            total_completed = sum(log['completed_tasks'] for log in log_analysis.values() if 'completed_tasks' in log)
            total_failed = sum(log['failed_tasks'] for log in log_analysis.values() if 'failed_tasks' in log)
            
            print(f"  已完成任务: {total_completed}")
            print(f"  失败任务: {total_failed}")
            print(f"  错误数量: {total_errors}")
            
            if total_errors > 0:
                print("  最近错误:")
                for log_name, log_data in log_analysis.items():
                    if 'recent_errors' in log_data and log_data['recent_errors']:
                        print(f"    {log_name}:")
                        for error in log_data['recent_errors']:
                            print(f"      - {error.strip()}")
        else:
            print("  没有找到日志文件")
        
        print("=" * 80)
    
    def print_simple_status(self):
        """打印简单状态"""
        progress = self.analyze_progress()
        sessions = self.get_tmux_sessions()
        
        print(f"实验状态: {'🟢 运行中' if sessions else '🔴 已停止'}")
        print(f"总体进度: {progress['total_generated']}/{progress['total_expected']} "
              f"({progress['total_progress']:.1f}%)")
        
        for category, data in progress['categories'].items():
            print(f"  {category}: {data['generated']}/{data['expected']} ({data['progress']:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="实验状态检查工具")
    parser.add_argument("--detailed", action="store_true", help="显示详细报告")
    args = parser.parse_args()
    
    checker = ExperimentChecker()
    
    if args.detailed:
        checker.print_detailed_report()
    else:
        checker.print_simple_status()

if __name__ == "__main__":
    main()

import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import os, re
import argparse


# Download models
#snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="/home/Wind645/Wan2.1-T2V-1.3B")
def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    """
    在 output_dir 下查找所有 prefixN.ext，返回下一个可用的完整路径。
    例如已有 video1.mp4、video2.mp4，则返回 results/video3.mp4
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")
def main(args):
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "/root/autodl-tmp/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "/root/autodl-tmp/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "/root/autodl-tmp/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)


    video = VideoData("/home/Wind645/code/diffsynth4MT/ref_videos/car_turn_41.mp4", height=480, width=832)
    # Text-to-video
    video = pipe(
        prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
        #prompt="纪实摄影风格，一只活泼的熊猫在绿茵茵的草地上迅速奔跑。熊猫毛色黑白分明，四只爪子灵活地在草地上踩踏，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉熊猫奔跑时的动感和四周草地的生机。中景侧面移动视角。",
        #prompt = "纪实摄影风格,一头雄壮的狮子,鬃毛呈现出深棕色,带有金色光泽,正在沙漠中奔跑.狮子的体态矫健,肌肉线条紧绷而流畅,四肢强健有力,步伐矫健而迅猛,展现出一种王者的威严与速度感.狮子的鬃毛在风中微微飘动,仿佛在诉说着它的力量与自由.背景是广袤的沙漠：近处沙丘纹理粗糙,风蚀痕迹清晰可见；远处沙丘连绵至地平线,与炽白的天空交融.阳光近乎垂直照射,空气因高温微微扭曲,沙地上留下了狮子奔跑时的爪印,延伸至画面外.中景,平视角度,展现了炎热的户外狩猎氛围.",
        #prompt = "纪实摄影风格，一只白色毛发的小猫，在绿茵茵的草地上迅速奔跑。小猫毛色洁白，四只爪子灵活地在草地上踩踏，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小猫奔跑时的动感和四周草地的生机。",
        #prompt = "纪实摄影风格，一只灰色的狐狸在秋天的森林中穿行。狐狸的毛发柔软而蓬松，尾巴高高翘起，神情警觉而机敏。阳光透过树叶洒在它身上，使得毛发看上去格外温暖而明亮。背景是一片金黄色的落叶，偶尔有几朵野花点缀其间，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉狐狸在森林中穿行时的动感和四周自然环境的宁静。",
        #prompt="纪实摄影风格，一只灰褐色羽毛的雄鹰在蓝天中翱翔。鹰的翅膀展开，展现出强壮的肌肉线条，羽毛在阳光下闪烁着金属般的光泽。背景是一片蔚蓝的天空，偶尔有几朵白云飘过，远处隐约可见连绵的山脉。透视感鲜明，捕捉鹰在空中飞翔时的动感和四周天空的辽阔。中景侧面移动视角。",
        #prompt="纪实摄影风格​​，一架​​红色的小型螺旋桨飞机​​，配备黑色轮胎，正沿着蜿蜒的柏油跑道滑行起飞。机身线条​​紧凑流畅​​，机翼设计简洁利落，驾驶舱玻璃反射着天光。背景是郁郁葱葱的茂密森林，远处巍峨的山脉云雾缭绕，蓝天白云相映成趣.中景,平视角度,展现自然清新的户外驾驶氛围.",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        input_video=video,
        input_video_path = "/home/Wind645/code/diffsynth4MT/ref_videos/car_turn_41.mp4",
        seed=args.seed, 
        tiled=True,
        num_frames=41,
        sf=args.sf,
        test_latency=args.test_latency,
        latency_dir=args.latency_dir,
        mode=args.mode,
    )
    save_video(video, get_next_video_path(output_dir=args.output_dir), fps=15, quality=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo Text-to-Video Example")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output videos")
    #parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_tile", action="store_true", help="Use tiled processing for large videos")
    parser.add_argument("--sf", type=int, default=5, help="Spatial factor for AMF computation (default: 1)")
    parser.add_argument("--test_latency", action="store_true", help="Test latency of the model")
    parser.add_argument("--latency_dir", type=str, default=None, help="Directory to save latency logs")
    parser.add_argument("--mode", type=str, default=None, choices=['No_transfer', 'AMF', 'effi_AMF'],help="Mode for the video generation, e.g., 'No_transfer'")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(args)
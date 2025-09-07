import types
from typing import Callable, Tuple
import math
from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import gc
import json

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_motion_controller import WanMotionControllerModel
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder', 'motion_controller', 'vace']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        self.vace = model_manager.fetch_model("wan_video_vace")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, end_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(self.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if self.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, self.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
    
    
    def encode_control_video(self, control_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        control_video = self.preprocess_images(control_video)
        control_video = torch.stack(control_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
        latents = self.encode_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
        return latents
    
    
    def prepare_controlnet_kwargs(self, control_video, num_frames, height, width, clip_feature=None, y=None, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        if control_video is not None:
            control_latents = self.encode_control_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if clip_feature is None or y is None:
                clip_feature = torch.zeros((1, 257, 1280), dtype=self.torch_dtype, device=self.device)
                y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=self.torch_dtype, device=self.device)
            else:
                y = y[:, -16:]
            y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}
    
    
    def prepare_motion_bucket_id(self, motion_bucket_id):
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=self.torch_dtype, device=self.device)
        return {"motion_bucket_id": motion_bucket_id}
    
    
    def prepare_vace_kwargs(
        self,
        latents,
        vace_video=None, vace_mask=None, vace_reference_image=None, vace_scale=1.0,
        height=480, width=832, num_frames=81,
        seed=None, rand_device="cpu",
        tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        if vace_video is not None or vace_mask is not None or vace_reference_image is not None:
            self.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=self.torch_dtype, device=self.device)
            else:
                vace_video = self.preprocess_images(vace_video)
                vace_video = torch.stack(vace_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            if vace_mask is None:
                vace_mask = torch.ones_like(vace_video)
            else:
                vace_mask = self.preprocess_images(vace_mask)
                vace_mask = torch.stack(vace_mask, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            inactive = vace_video * (1 - vace_mask) + 0 * vace_mask
            reactive = vace_video * vace_mask + 0 * (1 - vace_mask)
            inactive = self.encode_video(inactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            reactive = self.encode_video(reactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = self.preprocess_images([vace_reference_image])
                vace_reference_image = torch.stack(vace_reference_image, dim=2).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = self.encode_video(vace_reference_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
                
                noise = self.generate_noise((1, 16, 1, latents.shape[3], latents.shape[4]), seed=seed, device=rand_device, dtype=torch.float32)
                noise = noise.to(dtype=self.torch_dtype, device=self.device)
                latents = torch.concat((noise, latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return latents, {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return latents, {"vace_context": None, "vace_scale": vace_scale}
     
    # 编译优化，可选
    #@torch.compile
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_AMF(self, Q, K, sf, tau=3.0):
        """
        计算注意力运动流 (AMF)

        参数:
            Q (torch.Tensor): 查询矩阵，形状为 (F, S, D_h)
            K (torch.Tensor): 键矩阵，形状为 (F, S, D_h)
            tau (float): 温度参数

        返回:
            AMF (list): 注意力运动流，包含所有帧对的块位移矩阵
        """
        Q = Q.to('cuda')  # 全放在CUDA上
        K = K.to('cuda')
        f, h, w, D_h = Q.shape
        AMF = []
        Q = Q.view(f, h * w, D_h)  # 将 Q 重塑为 (F, S, D_h)
        K = K.view(f, h * w, D_h)  # 将 K 重塑为 (F, S, D_h)
        S = h * w
               
        # 预先计算 u 和 v 坐标
        u = torch.arange(h*w, device='cuda').unsqueeze(0) // w
        v = torch.arange(h*w, device='cuda').unsqueeze(0) % w
        
        #for i in range(f-sf):
        for i in range(f):
            #for j in range(i, i + sf):
            for j in range(f):
                # 计算跨帧注意力矩阵
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    A_ij = torch.matmul(Q[i], K[j].transpose(-1, -2)) / (D_h ** 0.5)

                    A_ij = F.softmax(A_ij * tau, dim=-1)  # 应用 softmax

                    u_j_approx = torch.sum(A_ij * (torch.arange(S, device='cuda') // w).unsqueeze(0), dim=-1)
                    v_j_approx = torch.sum(A_ij * (torch.arange(S, device='cuda') % w).unsqueeze(0), dim=-1)

                # 计算块位移
                delta_u = u_j_approx - u
                # print("delta_u requires_grad:", delta_u.requires_grad)
                delta_v = v_j_approx - v

                # 构建块位移矩阵
                Delta_ij = torch.stack([delta_u, delta_v], dim=-1)  # 形状为 (S, 2)
                AMF.append(Delta_ij)
        AMF = torch.stack(AMF)
        return AMF
        
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_tile_AMF(self, Q, K, sf, l=21, tau=3.0, tile=(3, 4)):
        """
        Compute the tile-based attention motion flow (AMF).

        Parameters:
            Q (torch.Tensor): Query matrix, shape (F, S, D_h)
            K (torch.Tensor): Key matrix, shape (F, S, D_h)
            sf (int): The farthest frame distance to consider
            tau (float): Temperature parameter

        Returns:
            AMF (list): Attention motion flow containing block displacement matrices for all frame pairs
        """
        #Q = Q.to('cuda')
        #K = K.to('cuda')
        f, h, w, D_h = K.shape
        Q = Q.view(f, h * w, D_h)  # 将 Q 重塑为 (F, S, D_h)
        K = K.view(f, h * w, D_h)  # 将 K 重塑为 (F, S, D_h)
        AMF = []
        #AMF = [[] for _ in range(sf)]
        S = h * w
        half_l = l // 2
        tile_h, tile_w = tile
        num_tiles_h = h // tile_h
        num_tiles_w = w // tile_w
        if not self.indices_computed:
            self.compute_indices(h, w, tile_h, tile_w, l, half_l, num_tiles_h, num_tiles_w, D_h)
        track_loss = []
        for i in range(f):
            K_wind = []
            for j in range(i, min(f, i + sf)):
                if j == i:
                    # 计算跨帧注意力矩阵
                    A_ij = torch.matmul(Q[i], K[j].transpose(-1, -2)) / (D_h ** 0.5)
                    A_ij = F.softmax(A_ij * tau, dim=-1)  # 应用 softmax

                    u_j_approx = torch.sum(A_ij * self.u, dim=-1)
                    v_j_approx = torch.sum(A_ij * self.v, dim=-1)

                    # 计算块位移
                    delta_u = u_j_approx - self.u
                    delta_v = v_j_approx - self.v

                    Delta_ij = torch.stack([delta_u, delta_v], dim=-1)
                    Delta_ij = Delta_ij.squeeze(0)  
                    AMF.append(Delta_ij)
                    #AMF[0].append(Delta_ij)  # 将结果添加到对应的列表中
                else:
                    # extract windows
                    Q_frame = Q[i]
                    K_frame = K[j].transpose(-1, -2)
                    
                    #u_j_approx, v_j_approx = self.compute_displacement(Q_frame, K_frame, u, v, linear_indices, D_h, tau)
                    Q_for_cal = torch.gather(Q_frame, 0, self.linear_indices)  # (h//tile_h * w//tile_w, D_h)
                    A_ij = torch.matmul(Q_for_cal, K_frame) / (D_h ** 0.5)
                    A_ij = F.softmax(A_ij * tau, dim=-1)  # 应用 softmax
                    u_j_approx = torch.sum(A_ij * self.u, dim=-1)
                    v_j_approx = torch.sum(A_ij * self.v, dim=-1)
                    
                    window_h = u_j_approx.view(num_tiles_h,num_tiles_w).unsqueeze(-1).unsqueeze(-1) + self.h_offsets
                    window_w = v_j_approx.view(num_tiles_h,num_tiles_w).unsqueeze(-1).unsqueeze(-1) + self.w_offsets
                    window_h = window_h.clamp(half_l, h - half_l - 1).long()
                    window_w = window_w.clamp(half_l, w - half_l - 1).long()
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        # extract K
                        linear_indexes = window_h * w + window_w  # 计算线性索引
                        linear_indexes_flat = linear_indexes.view(num_tiles_h*num_tiles_w, l*l)  # (h//tile_h * w//tile_w, l*l)
                        indices_expanded = linear_indexes_flat.unsqueeze(-1).expand(-1, -1, D_h).transpose(-1, -2)  # (hs x ws, l*l, D_h)
                        K_expanded = K[j].unsqueeze(1).expand(-1, l*l, -1).transpose(-1, -2)  # (h x w, l*l, D_h)
                        Q_expanded = Q[i].reshape(num_tiles_h, tile_h, num_tiles_w, tile_w, D_h).permute(0, 2, 1, 3, 4).reshape(-1, tile_h * tile_w, D_h)
                        
                        #delta_u, delta_v = self.compute_displacement_in_window(K_expanded, Q_expanded, posi_u_in_window, posi_v_in_window, indices_expanded, D_h, tau, S, half_l)
                        K_gathered = torch.gather(K_expanded, 0, indices_expanded)  # (hs x ws, l*l, D_h)
                        
                        K_window_mean = K_gathered.mean(dim=1)  # (hs x ws, D_h)
                        K_wind.append(K_window_mean)  # 保存当前窗口的 K 矩阵
                        
                        A_ij = (torch.matmul(Q_expanded, K_gathered) / (D_h ** 0.5)) # (hs x ws, tile_h*tile_w, l*l)                   
                        A_ij = F.softmax(A_ij.reshape(S, -1) * tau, dim=-1)
                        # calculate the displacement from the center of the window
                        u_j_approx = torch.sum(A_ij * self.posi_u_in_window, dim=-1) # (S,)
                        v_j_approx = torch.sum(A_ij * self.posi_v_in_window, dim=-1) # (S,)
                        delta_u = u_j_approx - self.orig_u
                        delta_v = v_j_approx - self.orig_v
                        # 构建块位移矩阵
                        Delta_ij = torch.stack([delta_u, delta_v], dim=-1)
                    AMF.append(Delta_ij)
                    #AMF[j - i].append(Delta_ij)  # 将结果添加到对应的列表中
            if len(K_wind) != 0:
                # 用 float32 做归约，避免 BF16 溢出；并清理 NaN/Inf
                K_tensor = torch.stack(K_wind, dim=0).float()  # (sf, hs*ws, D_h)
                diff = K_tensor[1:] - K_tensor[:-1]            # (sf-1, hs*ws, D_h)
                l2 = torch.linalg.vector_norm(diff, ord=2, dim=-1)  # (sf-1, hs*ws)
                l2_mean = l2.mean(dim=-1).mean()
                if not l2_mean.isnan():
                    track_loss.append(l2_mean)
        AMF = torch.stack(AMF)   
        tracking_loss = torch.stack(track_loss).mean()    
        #AMF = [torch.stack(amf) for amf in AMF]  # 将每个列表转换为张量          
        return AMF, tracking_loss
    
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_tile_amf_loss(self, amf_ref_orin, amf_gen):
        """
        Compute the loss between the original AMF and the generated AMF.
        
        param:
            amf_ref_orin (torch.Tensor): The original AMF tensor, shape (B, S, 2).
            amf_gen (torch.Tensor): The generated AMF tensor, shape (B, S, 2).
            
        Output:
            loss (torch.Tensor): The computed loss, averaged over the batch size.
        2-norm loss is used.
        """
        assert amf_ref_orin.shape == amf_gen.shape, "AMF reference and generated must have the same length"
        amf_ref_orin = amf_ref_orin.view(-1, 2 * amf_ref_orin.shape[1])
        amf_gen = amf_gen.view(-1, 2 * amf_gen.shape[1])
        diff = amf_ref_orin.detach() - amf_gen
        loss = ((torch.norm(diff, p=2, dim=-1) ** 2) * self.weights).mean()
        
        return loss  
    
    
    def compute_indices(self, h, w, tile_h, tile_w, l, half_l, num_tiles_h, num_tiles_w, D_h):
        u = torch.arange(h*w, device='cuda').unsqueeze(0) // w
        v = torch.arange(h*w, device='cuda').unsqueeze(0) % w
        
        posi_u_in_window = (torch.arange(l**2, device='cuda') // l).unsqueeze(0)
        posi_v_in_window = (torch.arange(l**2, device='cuda') % l).unsqueeze(0)
        
        #rows = torch.arange(0, h, tile_h, device='cuda').long()
        #cols = torch.arange(0, w, tile_w, device='cuda').long()
        rows = torch.arange(tile_h // 2, h + tile_h // 2, tile_h, device='cuda').long()
        cols = torch.arange(tile_w // 2, w + tile_w // 2, tile_w, device='cuda').long()
        
        h_offsets = torch.arange(-half_l, half_l + 1, device='cuda').view(1, 1, l, 1).expand(-1,-1,l,l)
        w_offsets = torch.arange(-half_l, half_l + 1, device='cuda').view(1, 1, 1, l).expand(-1,-1,l,l)
        
        orig_u = torch.arange(tile_h * tile_w, device='cuda').repeat(num_tiles_h * num_tiles_w) // tile_w
        orig_v = torch.arange(tile_h * tile_w, device='cuda').repeat(num_tiles_h * num_tiles_w) % tile_w
        #num_cals = f * sf - (1 + sf) * sf // 2
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing='ij')
        grid_row = grid_row.flatten() # (num_tiles_h * num_tiles_w,)
        grid_col = grid_col.flatten()  # (num_tiles_h * num_tiles
        linear_indices = grid_row * w + grid_col  
        linear_indices = linear_indices.view(-1) 
        linear_indices = linear_indices.unsqueeze(-1).expand(-1, D_h)
        self.u = u
        self.v = v
        #self.posi_u_in_window = posi_u_in_window
        #self.posi_v_in_window = posi_v_in_window
        # Sparsed
        mask = (torch.arange(l**2, device='cuda').unsqueeze(0) % 2 == 0).float()
        self.posi_u_in_window = posi_u_in_window * mask
        self.posi_v_in_window = posi_v_in_window * mask
        self.orig_u = orig_u
        self.orig_v = orig_v
        self.h_offsets = h_offsets
        self.w_offsets = w_offsets
        self.linear_indices = linear_indices
        self.indices_computed = True
      
            
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_amf_loss(self, amf_ref_orin, amf_gen):
        """
        Compute the loss between the original AMF and the generated AMF.
        
        param:
            amf_ref_orin (torch.Tensor): The original AMF tensor, shape (B, S, 2).
            amf_gen (torch.Tensor): The generated AMF tensor, shape (B, S, 2).
            
        Output:
            loss (torch.Tensor): The computed loss, averaged over the batch size.
        2-norm loss is used.
        """
        loss = torch.tensor(0.0, device=amf_gen[0].device) 
        
        diff = amf_ref_orin.detach() - amf_gen
        loss = torch.norm(diff, p=2) ** 2
        
        return loss / amf_ref_orin.shape[0]
    
    
    def clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  
        
    def guidance_step(self, 
                      latents: torch.Tensor, 
                      timestep: torch.Tensor,
                      prompt_emb_posi,
                      prompt_null,
                      image_emb,
                      extra_input,
                      noise, 
                      size_info, 
                      sf, # sf is the farthest frame distance to consider
                      mode, 
                      seed,
                      step_id,
                      interval: int=3
                      ) -> torch.Tensor:
        # Do the guidance step to optimize the latent representation.
        
        # Initialize the learning rate
        self.clean_memory()
        initial_lr = 0.003  
        final_lr = 0.002   
        total_steps = 10   # total optimization steps
        current_step = 0    # current optimization step
        
        if mode == 'AMF':
            initial_lr = 0.0003
            final_lr = 0.0002
        
        for i, block in enumerate(self.dit.blocks):  
            
            if i == 14:  
                self_attn = block.self_attn  
                self_attn.save_qk = True  # save the q_k and k_k for AMF computation
                break 
        
        optimized_latents = latents.clone().detach().requires_grad_(True)
        # optimized_x: copy the current latent representation and enable gradient computation
        optimizer = torch.optim.AdamW([optimized_latents], lr=initial_lr)
        # initialize the optimizer with the optimized_latents
        
        self.scale_range = np.linspace(0.007 , 0.004, 50)#[0.007 , 0.004],50,self.scale_range = np.linspace(config["scale_range"][0], config["scale_range"][1], len(self.guidance_schedule))
        
        with torch.no_grad(): # disable the gradient computation to calculate reference of the ref video
            noise = self.generate_noise(latents.shape, device=latents.device, dtype=torch.float32, seed=seed)
            noise = noise.to(dtype=self.torch_dtype, device=self.device)
            ref_latents = self.scheduler.add_noise(self.clean_latents, noise, timestep=timestep)
            _ = self.dit(ref_latents, timestep=timestep, preserve_space=True, size_info=size_info, **prompt_null, **image_emb, **extra_input)
            num_blocks = len(self.dit.blocks)
            print(f"Total number of blocks: {num_blocks}")
            for i, block in enumerate(self.dit.blocks):  # retrieve the index using enumerate
                if i == 14:  # The 15th block (index starts from 0)
                    self_attn = block.self_attn  # retrieve the WanSelfAttention instance
                    k_k = self_attn.k_reshape  # retrieve self.k_k
                    q_q = self_attn.q_reshape  # retrieve self.q_q
                    break  # quitting the loop after finding the 15th block
            if mode == 'AMF':
                amf = self.compute_AMF(q_q,k_k,sf=sf, tau=1.0)
                #amf = self.compute_AMF_1(q_q,k_k, tau=1.0)
            elif mode == 'effi_AMF':
                amf, _ = self.compute_tile_AMF(q_q,k_k,sf=sf, l=21, tau=1.0, tile=(3, 4))
            else:
                raise ValueError('Please set a valid mode to generate videos')
            
        # Compute AMF of the generated latent
        
        print(torch.cuda.memory_allocated() / 1024**2, "MB used")
        self.clean_memory()
        detached_prompt_emb_posi_new = {k: v.detach() if hasattr(v, 'detach') else v for k, v in prompt_emb_posi.items()}
        for j in tqdm(range(total_steps)):
            lr = initial_lr - (initial_lr - final_lr) * (current_step / total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    
                    self.dit.train()
                    #if j % interval == 0 or use_AMF:
                    #if j % interval == 0:
                    if j % interval == 0 or mode == 'AMF':
                        _ = self.dit(optimized_latents, timestep=timestep, size_info=size_info, preserve_space=True, **detached_prompt_emb_posi_new, **image_emb, **extra_input)

                        #print("OUT FORWARD")
                        for i, block in enumerate(self.dit.blocks):  
            
                            if i == 14:  
                                self_attn = block.self_attn  
                                k_k_new = self_attn.k_reshape  
                                q_q_new = self_attn.q_reshape 
                                break 
                        if mode == 'AMF': 
                            amf_new = self.compute_AMF(q_q_new,k_k_new,sf=sf, tau=1.0)
                            #amf_new = self.compute_AMF_1(q_q_new,k_k_new, tau=1.0)
                            loss = self.compute_amf_loss(amf, amf_new) / 1e3
                        elif mode == 'effi_AMF':
                            amf_new, track_loss = self.compute_tile_AMF(q_q_new,k_k_new,sf=sf, l=21, tau=1.0, tile=(3, 4))
                            #loss = self.compute_amf_loss(amf, amf_new)
                            amf_loss = self.compute_tile_amf_loss(amf, amf_new)
                            loss =  5*amf_loss + track_loss
                    else:
                        loss = (optimized_latents * self.cached_grad).sum()
                        #amf_new = self.compute_AMF(q_q_new,k_k_new,sf=sf, tau=1.0)
                        #loss = self.compute_amf_loss(amf, amf_new)

                    loss.backward()       
                    self.cached_grad = optimized_latents.grad.clone().detach()  
                    #print("Gradient norm:", torch.norm(optimized_latents.grad))
                    optimizer.step()

                    self.clean_memory()
            current_step = current_step + 1
            print("lr",lr)
        return optimized_latents
    
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        input_video=None,
        control_video=None,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        denoising_strength=0.75,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.3,
        num_inference_steps=50,
        sigma_shift=6.8,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        sf=4,
        test_latency=False,
        latency_dir=None,
        mode=None,
    ):
        self.indices_computed = False
        self.indices_expanded = []
        if mode is None:
            print('You did not specify the mode of transfer, we use efficient AMF mode by default.')
            mode = 'effi_AMF'
        elif mode == 'AMF':
            print('You are using the AMF mode, which is more accurate but slower.')
        elif mode == 'effi_AMF':
            print('You are using the efficient AMF mode, which is faster but less accurate.')
        elif mode == 'No_transfer':
            print('You are using the No_transfer mode, which does not use any transfer method.')
        elif mode == 'MOFT':
            print('You are using the MOFT mode, which uses the MOtion FeaTure to transfer the motion.')
        else:
            raise ValueError('Please set a valid mode to generate videos')
        
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        # size_info
        latent_frames = (num_frames - 1) // 4 + 1
        size_info = {'tiled': tiled, 'tile_size': tile_size, 'frames': latent_frames}
        # This line is added to preserve size information of the latent, which will then be passed
        # to reshape the q, k
        
        weights = np.linspace(1, 0.8, num=sf)
        for i in range(1, latent_frames):
            if i + sf < latent_frames:        
                weights = np.concatenate([weights, np.linspace(1, 0.8, num=sf)], axis=0)
            else:
                weights = np.concatenate([weights, np.linspace(1, 0.8, num=sf)[:latent_frames-i]], axis=0)
        self.weights = torch.tensor(weights, dtype=torch.float32, device='cuda')
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        
            
        if input_video is not None:                
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            self.clean_latents = latents.clone().detach()
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
            #latents = noise
        else:
            latents = noise
          
        #self.orin_latents = latents
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_null = self.encode_prompt("", positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, end_image, num_frames, height, width, **tiler_kwargs)
        else:
            image_emb = {}
            
        # ControlNet
        if control_video is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.prepare_controlnet_kwargs(control_video, num_frames, height, width, **image_emb, **tiler_kwargs)
            
        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # VACE
        latents, vace_kwargs = self.prepare_vace_kwargs(
            latents, vace_video, vace_video_mask, vace_reference_image, vace_scale,
            height=height, width=width, num_frames=num_frames, seed=seed, rand_device=rand_device, **tiler_kwargs
        )
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()


        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event = torch.cuda.Event(enable_timing=True)

        #start_event.record()
        # Denoise
        
        
        self.load_models_to_device(["dit", "motion_controller", "vace"])

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            i_for_guidance = 0
            if test_latency:
                start_event_guidance = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                start_event_gen = torch.cuda.Event(enable_timing=True)
                end_event_gen = torch.cuda.Event(enable_timing=True)
                end_event_guidance = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                
                start_event_gen.record()
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                if mode != 'No_transfer' and i_for_guidance < 20:
                    if test_latency:
                        start_event_guidance[i_for_guidance].record()
                    
                        latents = self.guidance_step(latents, timestep,prompt_emb_posi,prompt_null,image_emb,extra_input,noise,size_info, 
                                                 sf=sf, mode=mode, seed=seed, interval=3, step_id=progress_id)
                        end_event_guidance[i_for_guidance].record()
                    else:
                        latents = self.guidance_step(latents, timestep,prompt_emb_posi,prompt_null,image_emb,extra_input,noise,size_info,
                                                 sf=sf, mode=mode, seed=seed, interval=3, step_id=progress_id)
                    i_for_guidance += 1

                # Inference
                noise_pred_posi = model_fn_wan_video(
                    self.dit, motion_controller=self.motion_controller, vace=self.vace,
                    x=latents, timestep=timestep, size_info=size_info,
                    **prompt_emb_posi, **image_emb, **extra_input,
                    **tea_cache_posi, **usp_kwargs, **motion_kwargs, **vace_kwargs,
                )
                if cfg_scale != 1.0:
                    noise_pred_nega = model_fn_wan_video(
                        self.dit, motion_controller=self.motion_controller, vace=self.vace,
                        x=latents, timestep=timestep, size_info=size_info,
                        **prompt_emb_nega, **image_emb, **extra_input,
                        **tea_cache_nega, **usp_kwargs, **motion_kwargs, **vace_kwargs,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            self.clean_memory()
        if test_latency:     
            end_event_gen.record()
            torch.cuda.synchronize()  # 等待所有CUDA操作完成
            time_guidance = [start_event_guidance[i].elapsed_time(end_event_guidance[i]) / 1000 for i in range(10)]
            time_gen = start_event_gen.elapsed_time(end_event_gen) / 1000
            results = {
                "seed": seed,
                "time_guidance": time_guidance,
                "time_gen": time_gen,
            }
            with open(latency_dir, "a") as f:
                json.dump(results, f, ensure_ascii=False)
                f.write('\n') 
        if vace_reference_image is not None:
            latents = latents[:, :, 1:]
        
        del prompt_emb_posi, prompt_null, prompt_emb_nega
        #del self.dit, self.motion_controller, self.vace
        self.clean_memory()
        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    size_info: dict = None,
    **kwargs,
):
    return_intermediates = kwargs.pop("return_intermediates", False)
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = dit.patchify(x)
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    if return_intermediates:
        intermediates = []
    else:
        for block_id, block in enumerate(dit.blocks):
            x = block(x, context, t_mod, freqs, size_info)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                x = x + vace_hints[vace.vace_layers_mapping[block_id]] * vace_scale
            if return_intermediates:
                intermediates.append(x)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    if not return_intermediates:
        return x
    else:
        return x, intermediates
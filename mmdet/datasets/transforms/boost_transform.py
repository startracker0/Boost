import numpy as np
import torch
import math
from einops import rearrange
from opt_einsum import contract

import torch.nn.functional as F
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import RandAugment

import copy
import pynvml
import cv2
@TRANSFORMS.register_module()
class BoostTransform(BaseTransform):
    def __init__(self, kernel_size, sigma, groups, phases, sigma_min=0., granularity=64,decay=0):
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.kernels_size_candidates = torch.tensor([float(i) for i in range(self.kernel_size, self.kernel_size + 2, 2)])
        self.sigma_min = sigma_min
        self.sigma_max = sigma
        
        img_size = 1024
        _x = np.linspace(-img_size / 2, img_size / 2, img_size)
        self._x, self._y = np.meshgrid(_x, _x, indexing='ij')

        self.groups = groups
        self.num_groups = len(groups)
        self.freqs = [f / img_size for f in groups]

        self.phase_range = phases
        self.num_phases = granularity
        self.phases = -np.pi * np.linspace(phases[0], phases[1], num=granularity)

        f_cut=1
        phase_cut=1
        min_str=0
        mean_str=5
        
        self.f_cut = f_cut
        self.phase_cut = phase_cut

        self.min_str = min_str
        self.mean_str = mean_str

        self.eps_scale = img_size / 32
        self.decay = decay
        
    
    def transform(self, results):
        img = results['img'].copy()
        self._sample_params()
        self.device = results['gt_bboxes'].device

        img = self.randaug(img)
        filtered_img = self.filter(img)
        
        image = self.fourier(filtered_img)
      
        results['img'] = image
        
        return results
    
    
    def filter(self, img):
        img = img / 255.0

        img = np.transpose(img, (2, 0, 1))  # h w c -> c h w

        delta = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        delta[center, center] = 1.0

        conv_weight = self.sigma * np.random.randn(self.kernel_size, self.kernel_size) + delta

        filtered_img = np.zeros_like(img)
        
        for i in range(img.shape[0]):
            filtered_img[i] = self.apply_conv2d(img[i], conv_weight, padding=1)
        
        filtered_img = np.abs(filtered_img)
        
        # Normalize filtered_img to [0, 1] range
        min_val = filtered_img.min(axis=(1, 2), keepdims=True)
        max_val = filtered_img.max(axis=(1, 2), keepdims=True)
        filtered_img = (filtered_img - min_val) / (max_val - min_val + 1e-5)
      
        # Deal with NaN values
        filtered_img[np.isnan(filtered_img)] = 1

        # Clamp values to [0, 1]
        filtered_img = np.clip(filtered_img, 0., 1.)

        # Reshape back to the initial shape
        #filtered_img = rearrange(filtered_img[0], "c h w -> h w c")
        filtered_img = np.transpose(filtered_img, (1, 2, 0))
        filtered_img = (filtered_img * 255).astype(np.uint8)
        
        return filtered_img
    
    
    def get_gpu_info(self):
        # 初始化 pynvml
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            print(f"Failed to initialize pynvml: {e}")
            return []

        # 获取 GPU 数量
        device_count = pynvml.nvmlDeviceGetCount()

        gpu_info = []

        for i in range(device_count):
            # 获取 GPU 设备句柄
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            except pynvml.NVMLError as e:
                print(f"Failed to get handle for GPU {i}: {e}")
                continue

            # 获取 GPU 名称
            gpu_name = pynvml.nvmlDeviceGetName(handle)

            # 获取 GPU 显存信息
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = memory_info.total / 1024**2  # 转换为 MB
            memory_used = memory_info.used / 1024**2    # 转换为 MB
            memory_free = memory_info.free / 1024**2    # 转换为 MB

            # 获取 GPU 使用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu  # GPU 使用率（%）
            memory_utilization = utilization.memory  # 显存使用率（%）

            # 获取 GPU 温度
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError as e:
                print(f"Failed to get temperature for GPU {i}: {e}")
                temperature = None

            # 组织信息
            gpu_info.append({
                'GPU': i,
                'Name': gpu_name,  # 直接使用字符串
                'Memory Total (MB)': memory_total,
                'Memory Used (MB)': memory_used,
                'Memory Free (MB)': memory_free,
                'Memory Utilization (%)': memory_utilization,
                'GPU Utilization (%)': gpu_utilization,
                'Temperature (C)': temperature
            })

        # 清理 pynvml
        pynvml.nvmlShutdown()

        return gpu_info
    
    def get_gpu_with_max_free_memory(self):
        gpu_info = self.get_gpu_info()
        if gpu_info:
            # 找到显存剩余最大（Memory Free）GPU
            max_free_gpu = max(gpu_info, key=lambda x: x['Memory Free (MB)'])
            return max_free_gpu
        else:
            return None
    
    def apply_conv2d(self, img, kernel, padding=0):
        # gpu_with_max_free_memory = self.get_gpu_with_max_free_memory()
        # if gpu_with_max_free_memory != None:
        #     best_gpu_index = gpu_with_max_free_memory['GPU']
        #     device = torch.device(f'cuda:{best_gpu_index}')
        # else:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Apply padding
        img_tensor = F.pad(img_tensor, (padding, padding, padding, padding), mode='constant', value=0)

        # Apply convolution
        output = F.conv2d(img_tensor, kernel_tensor)

        return output.squeeze().cpu().numpy()
        
    
    def fourier(self,img):
        img = img / 255.0
        init_shape = img.shape

        if len(img.shape) < 4:
            img = rearrange(img, 'h w c -> () c h w')

        b, c, h, w = img.shape
        x_array = np.array(img, dtype=np.float32)
        

        freqs, phases, num_f, num_p = self.sample_f_p(b, c)
        strengths = np.random.exponential(1 / self.mean_str, phases.shape) + self.min_str
        aug = self.apply_fourier_aug(freqs, phases, strengths, x_array)
        
        aug = aug.cpu().numpy() 
        
        x_array = rearrange(x_array[0], "c h w -> h w c")
        aug = rearrange(aug[0], "c h w -> h w c")
        
        if self.decay > 0:
            aug = self.apply_gaussian_decay(aug)
        
        fourier_img = np.clip(x_array + aug, 0, 1)
        
        fourier_img = np.clip((fourier_img * 255).astype(np.uint8), 0, 255)
        
        return fourier_img.reshape(init_shape)
    
    def randaug(self,img):
        color_space = [
            [dict(type='mmdet.ColorTransform')],
            [dict(type='mmdet.AutoContrast')],
            [dict(type='mmdet.Equalize')],
            [dict(type='mmdet.Sharpness')],
            [dict(type='mmdet.Posterize')],
            [dict(type='mmdet.Solarize')],
            [dict(type='mmdet.Color')],
            [dict(type='mmdet.Contrast')],
            [dict(type='mmdet.Brightness')]
        ]
        
        # transform_color = RandAugment(aug_space=color_space, aug_num=1)
        transform_color = RandAugment(aug_space=color_space, aug_num=1)
        
        light_aug_img = dict(img = img)
        light_aug_img = transform_color(copy.deepcopy(light_aug_img))
        img = light_aug_img['img']
        
        return img
    
    
    def sample_f_p(self, b, c):
        f_cut = self.f_cut
        p_cut = self.phase_cut

        freqs = np.array(self.freqs, dtype=np.float32)
        phases = np.array(self.phases, dtype=np.float32)

        f_s = freqs[np.random.randint(0, self.num_groups, (b, c, f_cut, 1))]

        p_s = phases[np.random.randint(0, self.num_phases, (b, c, f_cut, p_cut))]

        return f_s, p_s, f_cut, p_cut
    
    def apply_fourier_aug(self, freqs, phases, strengths, x):
        aug = contract(
            'b c f p, b c f p h w -> b c h w',
            strengths,
            self.gen_planar_waves(freqs, phases)
        )
        #print(strengths.shape)
        aug *= 1 / (self.f_cut * self.phase_cut)
        
        b, c, h, w = x.shape

        aug = torch.tensor(aug, dtype=torch.float32).to(self.device)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        aug = torch.nn.functional.interpolate(aug, size=(h, w), mode='bilinear', align_corners=False)
        # apply_fourier_aug = torch.clamp(x + aug, 0, 1)
        
        return aug
    
    def apply_gaussian_decay(self, aug):
        h=aug.shape[0]
        w=aug.shape[1]
        center_x, center_y =np.random.randint(0, h - 13),np.random.randint(0, w - 13)
        sigma_x, sigma_y = h / 6, w / 6
        
        # 创建坐标网格
        x = torch.arange(h).float() - center_x
        y = torch.arange(w).float() - center_y
        X, Y = torch.meshgrid(x, y)
        
        # 计算高斯衰减矩阵
        gaussian_decay = torch.exp(-((X**2 / (2 * sigma_x**2)) + (Y**2 / (2 * sigma_y**2))))
        
        # 归一化到 [0, 1] 范围
        gaussian_decay = (gaussian_decay - gaussian_decay.min()) / (gaussian_decay.max() - gaussian_decay.min())
        
        gaussian_decay = (1-self.decay) + self.decay * gaussian_decay
        gaussian_decay = gaussian_decay.unsqueeze(-1) 
        gaussian_decay = gaussian_decay.cpu().numpy() 
        # 扩展到与 aug_resized 相同的维度
        # gaussian_decay = gaussian_decay.unsqueeze(0).unsqueeze(0)
        # print('aug',aug.shape)
        # print('decay',gaussian_decay.shape)
        decayed_aug = aug * gaussian_decay
        return decayed_aug
        
        
        

    def gen_planar_waves(self, freqs, phases):
        _x = torch.tensor(self._x, dtype=torch.float32)
        _y = torch.tensor(self._y, dtype=torch.float32)
        freqs = torch.tensor(freqs, dtype=torch.float32)
        phases = torch.tensor(phases, dtype=torch.float32)

        freqs, phases = rearrange(freqs, 'b c f p -> b c f p () ()'), rearrange(phases, 'b c f p -> b c f p () ()')
        _waves = torch.sin(
            2 * math.pi * freqs * (
                    _x * torch.cos(phases) + _y * torch.sin(phases)
            ) - math.pi / 4
        )
        _waves = _waves / torch.norm(_waves, dim=(-2, -1), keepdim=True)

        _waves = self.eps_scale * _waves
        return _waves
    
    
    def _sample_params(self):
        self.kernel_size = int(self.kernels_size_candidates[torch.multinomial(self.kernels_size_candidates, 1)].item())
        self.sigma = torch.FloatTensor([1]).uniform_(self.sigma_min, self.sigma_max).item()

    def __repr__(self):
        return self.__class__.__name__ + f"(sigma={self.sigma}, kernel_size={self.kernel_size},f={self.groups}, phases={self.phase_range},f_cut={self.f_cut}, p_cut={self.phase_cut}, min_str={self.min_str}, max_str={self.mean_str})"

    

"""
工具函数
图像转换、Base64编码、URL下载等
"""

import io
import base64
import requests
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将ComfyUI IMAGE tensor转换为PIL Image
    
    ComfyUI IMAGE格式: (B, H, W, C), float32, 0-1范围
    
    Args:
        tensor: 形状为 (B, H, W, C) 的tensor，取第一张图
        
    Returns:
        PIL.Image对象
    """
    # 确保在CPU上
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    
    # 取第一张图像
    if len(tensor.shape) == 4:
        img_np = tensor[0].numpy()
    else:
        img_np = tensor.numpy()
    
    # 转换到0-255范围
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    # 创建PIL Image
    if img_np.shape[2] == 4:
        mode = "RGBA"
    elif img_np.shape[2] == 3:
        mode = "RGB"
    else:
        mode = "L"
    
    return Image.fromarray(img_np, mode=mode)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    将PIL Image转换为ComfyUI IMAGE tensor
    
    Args:
        image: PIL.Image对象
        
    Returns:
        形状为 (1, H, W, C) 的tensor，float32, 0-1范围
    """
    # 确保是RGB或RGBA
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGB")
    
    # 转换为numpy数组
    img_np = np.array(image).astype(np.float32) / 255.0
    
    # 添加batch维度
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    
    return img_tensor


def tensor_to_base64(tensor: torch.Tensor, format: str = "PNG") -> str:
    """
    将ComfyUI IMAGE tensor转换为Base64编码字符串
    
    Args:
        tensor: ComfyUI IMAGE tensor
        format: 图像格式，如 "PNG", "JPEG"
        
    Returns:
        带MIME类型前缀的Base64字符串，如 "data:image/png;base64,xxxxx"
    """
    # 转换为PIL Image
    pil_image = tensor_to_pil(tensor)
    
    # 保存到内存缓冲区
    buffer = io.BytesIO()
    
    # 如果是RGBA且格式不支持透明，转换为RGB
    if pil_image.mode == "RGBA" and format.upper() == "JPEG":
        # 创建白色背景
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background
    
    pil_image.save(buffer, format=format.upper())
    
    # 编码为Base64
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # 添加MIME类型前缀
    mime_type = f"image/{format.lower()}"
    if format.upper() == "JPEG":
        mime_type = "image/jpeg"
    
    return f"data:{mime_type};base64,{base64_data}"


def mask_tensor_to_base64(mask_tensor: torch.Tensor, format: str = "PNG") -> str:
    """
    将ComfyUI MASK tensor转换为Base64编码的黑白图像
    
    ComfyUI MASK格式: (B, H, W), float32, 0-1范围
    白色(1.0)表示要编辑的区域，黑色(0.0)表示保留区域
    
    Args:
        mask_tensor: ComfyUI MASK tensor
        format: 图像格式
        
    Returns:
        带MIME类型前缀的Base64字符串
    """
    # 确保在CPU上
    if mask_tensor.device.type != "cpu":
        mask_tensor = mask_tensor.cpu()
    
    # 取第一张mask
    if len(mask_tensor.shape) == 3:
        mask_np = mask_tensor[0].numpy()
    else:
        mask_np = mask_tensor.numpy()
    
    # 转换到0-255范围（白色=要编辑的区域）
    mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)
    
    # 创建RGB图像（API需要RGB格式的黑白图像）
    mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
    
    # 创建PIL Image
    pil_image = Image.fromarray(mask_rgb, mode="RGB")
    
    # 保存到内存缓冲区
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format.upper())
    
    # 编码为Base64
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # 添加MIME类型前缀
    mime_type = f"image/{format.lower()}"
    
    return f"data:{mime_type};base64,{base64_data}"


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """
    从URL下载图像
    
    Args:
        url: 图像URL
        timeout: 超时时间（秒）
        
    Returns:
        PIL.Image对象
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    return Image.open(io.BytesIO(response.content))


def download_image_as_tensor(url: str, timeout: int = 30) -> torch.Tensor:
    """
    从URL下载图像并转换为ComfyUI tensor
    
    Args:
        url: 图像URL
        timeout: 超时时间（秒）
        
    Returns:
        ComfyUI IMAGE tensor
    """
    pil_image = download_image(url, timeout)
    return pil_to_tensor(pil_image)


def get_image_size(tensor: torch.Tensor) -> Tuple[int, int]:
    """
    获取图像尺寸
    
    Args:
        tensor: ComfyUI IMAGE tensor (B, H, W, C)
        
    Returns:
        (width, height)
    """
    if len(tensor.shape) == 4:
        _, h, w, _ = tensor.shape
    else:
        h, w, _ = tensor.shape
    return (w, h)


def validate_image_size(tensor: torch.Tensor, min_size: int = 512, max_size: int = 4096) -> Tuple[bool, str]:
    """
    验证图像尺寸是否符合API要求
    
    Args:
        tensor: ComfyUI IMAGE tensor
        min_size: 最小尺寸（像素）
        max_size: 最大尺寸（像素）
        
    Returns:
        (是否有效, 错误信息)
    """
    w, h = get_image_size(tensor)
    
    if w < min_size or h < min_size:
        return False, f"图像尺寸过小：{w}x{h}，最小要求 {min_size}x{min_size} 像素"
    
    if w > max_size or h > max_size:
        return False, f"图像尺寸过大：{w}x{h}，最大支持 {max_size}x{max_size} 像素"
    
    return True, ""


def resize_image_if_needed(
    tensor: torch.Tensor,
    min_size: int = 512,
    max_size: int = 4096
) -> Tuple[torch.Tensor, bool]:
    """
    如果图像尺寸不符合要求，则调整大小
    
    Args:
        tensor: ComfyUI IMAGE tensor
        min_size: 最小尺寸
        max_size: 最大尺寸
        
    Returns:
        (调整后的tensor, 是否调整过)
    """
    w, h = get_image_size(tensor)
    
    # 检查是否需要调整
    if min_size <= w <= max_size and min_size <= h <= max_size:
        return tensor, False
    
    # 计算新尺寸
    scale = 1.0
    if w < min_size or h < min_size:
        scale = max(min_size / w, min_size / h)
    elif w > max_size or h > max_size:
        scale = min(max_size / w, max_size / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 确保在范围内
    new_w = max(min_size, min(max_size, new_w))
    new_h = max(min_size, min(max_size, new_h))
    
    # 转换为PIL进行缩放
    pil_image = tensor_to_pil(tensor)
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 转回tensor
    return pil_to_tensor(pil_image), True


def concat_images_horizontal(tensors: list) -> torch.Tensor:
    """
    水平拼接多张图像（用于显示多个结果）
    
    Args:
        tensors: ComfyUI IMAGE tensor列表
        
    Returns:
        拼接后的tensor
    """
    if len(tensors) == 1:
        return tensors[0]
    
    # 转换为PIL
    pil_images = [tensor_to_pil(t) for t in tensors]
    
    # 计算总宽度和最大高度
    total_width = sum(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)
    
    # 创建新图像
    mode = "RGBA" if any(img.mode == "RGBA" for img in pil_images) else "RGB"
    combined = Image.new(mode, (total_width, max_height))
    
    # 拼接
    x_offset = 0
    for img in pil_images:
        # 垂直居中
        y_offset = (max_height - img.height) // 2
        combined.paste(img, (x_offset, y_offset))
        x_offset += img.width
    
    return pil_to_tensor(combined)


def batch_tensors(tensors: list) -> torch.Tensor:
    """
    将多个单张图像tensor合并为批次tensor
    
    Args:
        tensors: ComfyUI IMAGE tensor列表，每个形状为 (1, H, W, C)
        
    Returns:
        形状为 (N, H, W, C) 的批次tensor
    """
    if len(tensors) == 1:
        return tensors[0]
    
    return torch.cat(tensors, dim=0)


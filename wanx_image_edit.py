"""
通义万相2.1图像编辑 ComfyUI节点
支持10种图像编辑功能
"""

import torch
from typing import Tuple, Optional, List, Dict, Any

from .api_client import WanxImageEditClient, WanxAPIError, WanxImageEditFunction
from .utils import (
    tensor_to_base64,
    mask_tensor_to_base64,
    download_image_as_tensor,
    validate_image_size,
    resize_image_if_needed,
    batch_tensors,
)


# ============================================================
# API配置加载器节点
# ============================================================

class HaoranWanxAPILoader:
    """
    通义万相API配置加载器
    
    用于配置API连接参数，输出可复用的API配置对象。
    类似于 comfyui_LLM_party 的 API大语言模型加载器。
    """
    
    CATEGORY = "昊然"
    FUNCTION = "load_api"
    RETURN_TYPES = ("WANX_API",)
    RETURN_NAMES = ("api_config",)
    
    # 默认API端点
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["wanx2.1-imageedit"], {"default": "wanx2.1-imageedit"}),
                "base_url": ("STRING", {
                    "default": cls.DEFAULT_BASE_URL,
                    "multiline": False,
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
                }),
            }
        }
    
    def load_api(self, model_name: str, base_url: str, api_key: str) -> Tuple[Dict]:
        """
        加载API配置
        
        Returns:
            包含API配置的字典
        """
        # 验证API Key
        if not api_key.strip():
            import os
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            if not api_key:
                print("[Wanx API] 警告: API Key为空，请配置api_key参数或设置环境变量DASHSCOPE_API_KEY")
        
        config = {
            "model_name": model_name,
            "base_url": base_url.strip() or self.DEFAULT_BASE_URL,
            "api_key": api_key.strip(),
        }
        
        print(f"[Wanx API] 配置已加载")
        print(f"[Wanx API] 模型: {model_name}")
        print(f"[Wanx API] 端点: {config['base_url'][:50]}...")
        print(f"[Wanx API] Key: {'*' * 20 + api_key[-6:] if len(api_key) > 6 else '未配置'}")
        
        return (config,)


# ============================================================
# 主图像编辑节点（支持API配置输入）
# ============================================================


class HaoranWanxImageEdit:
    """
    通义万相2.1图像编辑节点
    
    支持的功能：
    - 全局风格化：将整张图像迁移至指定艺术风格
    - 局部风格化：仅对图像局部区域进行风格迁移
    - 指令编辑：通过文本指令增加或修改图像内容
    - 局部重绘：通过mask对指定区域进行编辑
    - 去文字水印：去除图像中的中英文字符或水印
    - 扩图：沿四个方向扩展画布并智能填充
    - 图像超分：提升图像清晰度并支持放大
    - 图像上色：将黑白/灰度图像转为彩色
    - 线稿生图：基于线稿生成新图像
    - 卡通形象生图：基于卡通形象生成新图像
    """
    
    CATEGORY = "昊然"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    
    # 功能选项（中文显示名称 -> API参数）
    FUNCTION_OPTIONS = {
        "全局风格化": "stylization_all",
        "局部风格化": "stylization_local",
        "指令编辑": "description_edit",
        "局部重绘": "description_edit_with_mask",
        "去文字水印": "remove_watermark",
        "扩图": "expand",
        "图像超分": "super_resolution",
        "图像上色": "colorization",
        "线稿生图": "doodle",
        "卡通形象生图": "control_cartoon_feature",
    }
    
    # 全局风格选项
    GLOBAL_STYLE_OPTIONS = [
        "法国绘本风格",
        "金箔艺术风格",
    ]
    
    # 局部风格选项
    LOCAL_STYLE_OPTIONS = [
        "冰雕 (ice)",
        "云朵 (cloud)",
        "花灯 (chinese festive lantern)",
        "木板 (wooden)",
        "青花瓷 (blue and white porcelain)",
        "毛茸茸 (fluffy)",
        "毛线 (weaving)",
        "气球 (balloon)",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 图像输入
                "图像": ("IMAGE",),
                "功能": (list(cls.FUNCTION_OPTIONS.keys()), {"default": "指令编辑"}),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "描述您想要的编辑效果..."
                }),
                
                # 通用参数
                "生成数量": ("INT", {"default": 1, "min": 1, "max": 4}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "使用随机种子": ("BOOLEAN", {"default": False}),
                "添加AI水印": ("BOOLEAN", {"default": False}),
                
                # 修改强度（全局风格化/指令编辑）
                "修改强度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # 扩图参数（最大2倍）
                "上方扩展": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
                "下方扩展": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
                "左侧扩展": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
                "右侧扩展": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
                
                # 超分参数
                "超分倍数": ("INT", {"default": 2, "min": 1, "max": 4}),
                
                # 线稿参数
                "涂鸦模式": ("BOOLEAN", {"default": False}),
                
                # 超时设置
                "超时秒数": ("INT", {"default": 180, "min": 30, "max": 600}),
                
                # 自动调整尺寸
                "自动调整尺寸": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # API配置（来自API加载器节点）
                "api_config": ("WANX_API",),
                # 直接输入API Key（如果不使用API加载器）
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "留空则使用api_config或环境变量"
                }),
                # 蒙版（局部重绘用）
                "蒙版": ("MASK",),
            }
        }
    
    def process(
        self,
        图像: torch.Tensor,
        功能: str,
        提示词: str,
        生成数量: int,
        随机种子: int,
        使用随机种子: bool,
        添加AI水印: bool,
        修改强度: float,
        上方扩展: float,
        下方扩展: float,
        左侧扩展: float,
        右侧扩展: float,
        超分倍数: int,
        涂鸦模式: bool,
        超时秒数: int,
        自动调整尺寸: bool,
        api_config: Optional[Dict] = None,
        api_key: Optional[str] = None,
        蒙版: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """执行图像编辑"""
        
        # 获取API功能名称
        function_name = self.FUNCTION_OPTIONS.get(功能)
        if not function_name:
            raise ValueError(f"不支持的功能: {功能}")
        
        print(f"\n{'='*50}")
        print(f"[Wanx] 通义万相2.1图像编辑")
        print(f"[Wanx] 功能: {功能} ({function_name})")
        print(f"[Wanx] 提示词: {提示词[:50]}..." if len(提示词) > 50 else f"[Wanx] 提示词: {提示词}")
        print(f"{'='*50}")
        
        # 验证必需参数
        self._validate_params(功能, 提示词, 蒙版)
        
        # 处理图像尺寸
        if 自动调整尺寸:
            图像, resized = resize_image_if_needed(图像)
            if resized:
                print(f"[Wanx] 图像已自动调整尺寸")
        else:
            valid, error_msg = validate_image_size(图像)
            if not valid:
                raise ValueError(error_msg)
        
        # 转换图像为Base64
        base_image_url = tensor_to_base64(图像, format="PNG")
        print(f"[Wanx] 图像已转换为Base64")
        
        # 处理蒙版（如果有）
        mask_image_url = None
        if function_name == "description_edit_with_mask" and 蒙版 is not None:
            mask_image_url = mask_tensor_to_base64(蒙版, format="PNG")
            print(f"[Wanx] 蒙版已转换为Base64")
        
        # 获取API Key（优先级：api_key参数 > api_config > 环境变量）
        final_api_key = None
        if api_key and api_key.strip():
            final_api_key = api_key.strip()
            print(f"[Wanx] 使用直接输入的API Key")
        elif api_config and api_config.get("api_key"):
            final_api_key = api_config["api_key"]
            print(f"[Wanx] 使用API配置节点的API Key")
        
        # 创建API客户端
        try:
            client = WanxImageEditClient(api_key=final_api_key)
        except ValueError as e:
            raise ValueError(str(e))
        
        # 获取最终提示词（可能使用默认值）
        final_prompt = self._get_prompt(功能, 提示词)
        if final_prompt != 提示词:
            print(f"[Wanx] 使用默认提示词: {final_prompt}")
        
        # 构建API参数
        api_params = {
            "function": function_name,
            "prompt": final_prompt,
            "base_image_url": base_image_url,
            "n": 生成数量,
            "watermark": 添加AI水印,
            "timeout": 超时秒数,
        }
        
        # 添加蒙版
        if mask_image_url:
            api_params["mask_image_url"] = mask_image_url
        
        # 添加随机种子
        if 使用随机种子 and 随机种子 > 0:
            api_params["seed"] = 随机种子
        
        # 根据功能添加特定参数
        if function_name in ["stylization_all", "description_edit"]:
            api_params["strength"] = 修改强度
        
        if function_name == "expand":
            api_params["top_scale"] = 上方扩展
            api_params["bottom_scale"] = 下方扩展
            api_params["left_scale"] = 左侧扩展
            api_params["right_scale"] = 右侧扩展
        
        if function_name == "super_resolution":
            api_params["upscale_factor"] = 超分倍数
        
        if function_name == "doodle":
            api_params["is_sketch"] = 涂鸦模式
        
        # 调用API
        try:
            result = client.edit_image(**api_params)
        except WanxAPIError as e:
            error_msg = self._format_error_message(e)
            raise RuntimeError(error_msg)
        
        # 下载结果图像
        results = result.get("results", [])
        if not results:
            raise RuntimeError("API返回结果为空")
        
        print(f"[Wanx] 获取到 {len(results)} 张结果图像")
        
        # 下载所有结果图像
        output_tensors = []
        for i, item in enumerate(results):
            url = item.get("url")
            if not url:
                continue
            
            print(f"[Wanx] 下载图像 {i+1}/{len(results)}...")
            tensor = download_image_as_tensor(url)
            output_tensors.append(tensor)
        
        if not output_tensors:
            raise RuntimeError("无法下载任何结果图像")
        
        # 合并为批次tensor
        output = batch_tensors(output_tensors)
        
        # 打印使用统计
        usage = result.get("usage", {})
        if usage:
            print(f"[Wanx] 使用统计: {usage}")
        
        print(f"[Wanx] 处理完成！")
        print(f"{'='*50}\n")
        
        return (output,)
    
    # 不需要提示词的功能（可留空，会使用默认值）
    OPTIONAL_PROMPT_FUNCTIONS = [
        "remove_watermark",    # 去文字水印：默认"去除图像中的文字"
        "super_resolution",    # 图像超分：默认"图像超分"
        "colorization",        # 图像上色：留空自动上色
    ]
    
    # 默认提示词
    DEFAULT_PROMPTS = {
        "remove_watermark": "去除图像中的文字",
        "super_resolution": "图像超分",
        "colorization": "",  # 上色可以留空
    }
    
    def _validate_params(self, 功能: str, 提示词: str, 蒙版: Optional[torch.Tensor]):
        """验证参数"""
        function_name = self.FUNCTION_OPTIONS.get(功能)
        
        # 某些功能必须提供提示词
        if function_name not in self.OPTIONAL_PROMPT_FUNCTIONS and not 提示词.strip():
            raise ValueError(f"功能「{功能}」需要提供提示词")
        
        # 局部重绘需要蒙版
        if function_name == "description_edit_with_mask" and 蒙版 is None:
            raise ValueError("功能「局部重绘」需要提供蒙版输入")
    
    def _get_prompt(self, 功能: str, 提示词: str) -> str:
        """获取提示词，如果为空则使用默认值"""
        function_name = self.FUNCTION_OPTIONS.get(功能)
        
        if 提示词.strip():
            return 提示词.strip()
        
        # 使用默认提示词
        return self.DEFAULT_PROMPTS.get(function_name, "")
    
    def _format_error_message(self, error: WanxAPIError) -> str:
        """格式化错误信息"""
        error_messages = {
            "InvalidApiKey": "API Key无效，请检查配置",
            "Arrearage": "账户欠费，请充值后重试",
            "DataInspectionFailed": "内容安全审核不通过，请修改输入内容",
            "TIMEOUT": "任务超时，请增加超时时间或稍后重试",
            "TASK_FAILED": "任务执行失败",
            "TASK_CANCELED": "任务已被取消",
            "NETWORK_ERROR": "网络请求失败，请检查网络连接",
        }
        
        friendly_msg = error_messages.get(error.code, error.message)
        
        return f"[Wanx错误] {friendly_msg}\n错误码: {error.code}\n详细信息: {error.message}"


# 提示词助手节点
class HaoranWanxPromptHelper:
    """
    通义万相提示词助手
    提供不同功能的提示词模板和建议
    """
    
    CATEGORY = "昊然"
    FUNCTION = "generate_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    
    # 提示词模板
    PROMPT_TEMPLATES = {
        "全局风格化": {
            "法国绘本风格": "转换成法国绘本风格",
            "金箔艺术风格": "转换成金箔艺术风格",
        },
        "局部风格化": {
            "冰雕": "把{对象}变成冰雕风格",
            "云朵": "把{对象}变成云朵风格",
            "花灯": "把{对象}变成花灯风格",
            "木板": "把{对象}变成木板风格",
            "青花瓷": "把{对象}变成青花瓷风格",
            "毛茸茸": "把{对象}变成毛茸茸风格",
            "毛线": "把{对象}变成毛线风格",
            "气球": "把{对象}变成气球风格",
        },
        "指令编辑": {
            "添加配饰": "给{对象}添加{配饰}",
            "修改颜色": "把{对象}的颜色修改为{颜色}",
            "更换服装": "将{对象}的衣服修改为{服装}",
        },
        "去文字水印": {
            "去除所有文字": "去除图像中的文字",
            "去除英文": "去除英文文字",
            "去除水印": "去除图像中的水印",
        },
        "扩图": {
            "场景描述": "{描述扩展后期望看到的完整场景}",
        },
        "图像超分": {
            "默认": "图像超分",
        },
        "图像上色": {
            "自动上色": "",
            "指定颜色": "{颜色描述}，{颜色描述}",
        },
        "线稿生图": {
            "场景描述": "{详细描述期望生成的图像内容}",
        },
        "卡通形象生图": {
            "动作场景": "卡通形象{动作描述}，{场景描述}",
        },
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "功能": (list(cls.PROMPT_TEMPLATES.keys()), {"default": "指令编辑"}),
                "模板": ("STRING", {"default": ""}),
                "自定义内容": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "输入要替换模板中{xxx}的内容，或完全自定义..."
                }),
            }
        }
    
    def generate_prompt(self, 功能: str, 模板: str, 自定义内容: str) -> Tuple[str]:
        """生成提示词"""
        templates = self.PROMPT_TEMPLATES.get(功能, {})
        
        if 自定义内容.strip():
            # 如果有自定义内容，直接返回
            return (自定义内容,)
        
        if 模板 and 模板 in templates:
            return (templates[模板],)
        
        # 返回第一个模板作为默认
        if templates:
            first_template = list(templates.values())[0]
            return (first_template,)
        
        return ("",)


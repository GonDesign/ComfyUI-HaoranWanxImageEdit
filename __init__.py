"""
ComfyUI-HaoranWanxImageEdit
通义万相2.1图像编辑 ComfyUI 插件

支持功能：
- 全局风格化：法国绘本风格、金箔艺术风格
- 局部风格化：冰雕、云朵、花灯、木板、青花瓷等8种风格
- 指令编辑：通过文本指令增加或修改图像内容
- 局部重绘：通过mask对指定区域进行精确编辑
- 去文字水印：去除图像中的中英文字符或水印
- 扩图：沿四个方向扩展画布并智能填充
- 图像超分：提升图像清晰度，支持1-4倍放大
- 图像上色：将黑白/灰度图像转为彩色
- 线稿生图：基于线稿或涂鸦生成新图像
- 卡通形象生图：基于卡通形象生成新图像

作者: Haoran
版本: 1.0.0
"""

from .wanx_image_edit import HaoranWanxAPILoader, HaoranWanxImageEdit, HaoranWanxPromptHelper

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "HaoranWanxAPILoader": HaoranWanxAPILoader,
    "HaoranWanxImageEdit": HaoranWanxImageEdit,
    "HaoranWanxPromptHelper": HaoranWanxPromptHelper,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "HaoranWanxAPILoader": "Haoran 万相API配置",
    "HaoranWanxImageEdit": "Haoran 通义万相图像编辑",
    "HaoranWanxPromptHelper": "Haoran 万相提示词助手",
}

# 版本信息
__version__ = "1.0.0"
__author__ = "Haoran"

# 导出
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]

# 初始化日志
print(f"[ComfyUI-HaoranWanxImageEdit] 插件已加载 v{__version__}")
print(f"[ComfyUI-HaoranWanxImageEdit] 节点列表:")
print(f"  - Haoran 万相API配置: 配置API Key和端点")
print(f"  - Haoran 通义万相图像编辑: 10种图像编辑功能")
print(f"  - Haoran 万相提示词助手: 提示词模板和建议")


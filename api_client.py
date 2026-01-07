"""
阿里云百炼 wanx2.1-imageedit API 客户端
封装异步任务提交、轮询查询、结果获取等功能
"""

import os
import time
import requests
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "PENDING"      # 任务排队中
    RUNNING = "RUNNING"      # 任务处理中
    SUCCEEDED = "SUCCEEDED"  # 任务执行成功
    FAILED = "FAILED"        # 任务执行失败
    CANCELED = "CANCELED"    # 任务已取消
    UNKNOWN = "UNKNOWN"      # 任务不存在或状态未知


class WanxImageEditFunction(Enum):
    """图像编辑功能枚举"""
    STYLIZATION_ALL = "stylization_all"                    # 全局风格化
    STYLIZATION_LOCAL = "stylization_local"                # 局部风格化
    DESCRIPTION_EDIT = "description_edit"                  # 指令编辑
    DESCRIPTION_EDIT_WITH_MASK = "description_edit_with_mask"  # 局部重绘
    REMOVE_WATERMARK = "remove_watermark"                  # 去文字水印
    EXPAND = "expand"                                      # 扩图
    SUPER_RESOLUTION = "super_resolution"                  # 图像超分
    COLORIZATION = "colorization"                          # 图像上色
    DOODLE = "doodle"                                      # 线稿生图
    CONTROL_CARTOON_FEATURE = "control_cartoon_feature"    # 参考卡通形象生图


class WanxAPIError(Exception):
    """API调用错误"""
    def __init__(self, code: str, message: str, request_id: str = None):
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(f"[{code}] {message}" + (f" (request_id: {request_id})" if request_id else ""))


class WanxImageEditClient:
    """
    通义万相2.1图像编辑API客户端
    
    使用示例:
        client = WanxImageEditClient(api_key="sk-xxx")
        result = client.edit_image(
            function="description_edit",
            prompt="给她戴上一副墨镜",
            base_image_url="https://example.com/image.jpg"
        )
    """
    
    # API端点
    BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
    SUBMIT_ENDPOINT = "/services/aigc/image2image/image-synthesis"
    TASK_ENDPOINT = "/tasks/{task_id}"
    
    # 默认配置
    DEFAULT_TIMEOUT = 180  # 默认超时时间（秒）
    DEFAULT_POLL_INTERVAL_FAST = 3  # 前30秒的轮询间隔
    DEFAULT_POLL_INTERVAL_SLOW = 5  # 30秒后的轮询间隔
    DEFAULT_FAST_POLL_DURATION = 30  # 快速轮询持续时间
    MAX_RETRIES = 3  # 最大重试次数
    
    def __init__(self, api_key: str = None):
        """
        初始化客户端
        
        Args:
            api_key: 阿里云百炼API Key，如果不提供则从环境变量DASHSCOPE_API_KEY获取
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API Key未配置！请通过参数传入或设置环境变量DASHSCOPE_API_KEY。\n"
                "获取API Key: https://bailian.console.aliyun.com/?tab=model#/api-key"
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
    
    def submit_task(
        self,
        function: str,
        prompt: str,
        base_image_url: str,
        mask_image_url: str = None,
        n: int = 1,
        seed: int = None,
        watermark: bool = False,
        strength: float = None,
        top_scale: float = None,
        bottom_scale: float = None,
        left_scale: float = None,
        right_scale: float = None,
        upscale_factor: int = None,
        is_sketch: bool = None,
    ) -> str:
        """
        提交图像编辑任务
        
        Args:
            function: 编辑功能
            prompt: 提示词
            base_image_url: 输入图像URL或Base64
            mask_image_url: 涂抹区域图像（局部重绘时需要）
            n: 生成图片数量 (1-4)
            seed: 随机种子
            watermark: 是否添加水印
            strength: 修改幅度 (0.0-1.0)
            top_scale/bottom_scale/left_scale/right_scale: 扩图比例
            upscale_factor: 超分倍数 (1-4)
            is_sketch: 是否为涂鸦模式
            
        Returns:
            task_id: 任务ID
        """
        # 构建请求体
        input_data = {
            "function": function,
            "prompt": prompt,
            "base_image_url": base_image_url,
        }
        
        # 局部重绘需要mask
        if mask_image_url:
            input_data["mask_image_url"] = mask_image_url
        
        # 构建parameters
        parameters = {"n": n}
        if seed is not None:
            parameters["seed"] = seed
        if watermark:
            parameters["watermark"] = True
        if strength is not None:
            parameters["strength"] = strength
        if top_scale is not None:
            parameters["top_scale"] = top_scale
        if bottom_scale is not None:
            parameters["bottom_scale"] = bottom_scale
        if left_scale is not None:
            parameters["left_scale"] = left_scale
        if right_scale is not None:
            parameters["right_scale"] = right_scale
        if upscale_factor is not None:
            parameters["upscale_factor"] = upscale_factor
        if is_sketch is not None:
            parameters["is_sketch"] = is_sketch
        
        request_body = {
            "model": "wanx2.1-imageedit",
            "input": input_data,
            "parameters": parameters,
        }
        
        # 发送请求
        url = f"{self.BASE_URL}{self.SUBMIT_ENDPOINT}"
        headers = {"X-DashScope-Async": "enable"}
        
        response = self._request_with_retry("POST", url, json=request_body, headers=headers)
        
        # 解析响应
        data = response.json()
        
        if "code" in data:
            raise WanxAPIError(
                code=data.get("code", "UNKNOWN"),
                message=data.get("message", "未知错误"),
                request_id=data.get("request_id")
            )
        
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise WanxAPIError(
                code="NO_TASK_ID",
                message="响应中未包含task_id",
                request_id=data.get("request_id")
            )
        
        print(f"[Wanx] 任务已提交: {task_id}")
        return task_id
    
    def query_task(self, task_id: str) -> Dict[str, Any]:
        """
        查询任务状态和结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息字典，包含status, results等
        """
        url = f"{self.BASE_URL}{self.TASK_ENDPOINT.format(task_id=task_id)}"
        response = self._request_with_retry("GET", url)
        
        data = response.json()
        
        if "code" in data and data["code"]:
            raise WanxAPIError(
                code=data.get("code", "UNKNOWN"),
                message=data.get("message", "未知错误"),
                request_id=data.get("request_id")
            )
        
        output = data.get("output", {})
        return {
            "task_id": output.get("task_id"),
            "task_status": output.get("task_status"),
            "results": output.get("results", []),
            "code": output.get("code"),
            "message": output.get("message"),
            "submit_time": output.get("submit_time"),
            "scheduled_time": output.get("scheduled_time"),
            "end_time": output.get("end_time"),
            "task_metrics": output.get("task_metrics", {}),
            "usage": data.get("usage", {}),
            "request_id": data.get("request_id"),
        }
    
    def wait_for_result(
        self,
        task_id: str,
        timeout: int = None,
        poll_interval_fast: float = None,
        poll_interval_slow: float = None,
    ) -> Dict[str, Any]:
        """
        等待任务完成并返回结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            poll_interval_fast: 前30秒的轮询间隔
            poll_interval_slow: 30秒后的轮询间隔
            
        Returns:
            任务结果
        """
        timeout = timeout or self.DEFAULT_TIMEOUT
        poll_interval_fast = poll_interval_fast or self.DEFAULT_POLL_INTERVAL_FAST
        poll_interval_slow = poll_interval_slow or self.DEFAULT_POLL_INTERVAL_SLOW
        
        start_time = time.time()
        poll_count = 0
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                raise WanxAPIError(
                    code="TIMEOUT",
                    message=f"任务超时（已等待{timeout}秒）"
                )
            
            # 查询任务状态
            result = self.query_task(task_id)
            status = result.get("task_status")
            poll_count += 1
            
            print(f"[Wanx] 轮询 #{poll_count}: 状态={status}, 耗时={elapsed:.1f}秒")
            
            if status == TaskStatus.SUCCEEDED.value:
                print(f"[Wanx] 任务完成！总耗时: {elapsed:.1f}秒")
                return result
            
            elif status == TaskStatus.FAILED.value:
                error_code = result.get("code", "TASK_FAILED")
                error_msg = result.get("message", "任务执行失败")
                raise WanxAPIError(
                    code=error_code,
                    message=error_msg,
                    request_id=result.get("request_id")
                )
            
            elif status == TaskStatus.CANCELED.value:
                raise WanxAPIError(
                    code="TASK_CANCELED",
                    message="任务已被取消"
                )
            
            elif status in [TaskStatus.PENDING.value, TaskStatus.RUNNING.value]:
                # 根据已等待时间选择轮询间隔
                if elapsed < self.DEFAULT_FAST_POLL_DURATION:
                    interval = poll_interval_fast
                else:
                    interval = poll_interval_slow
                time.sleep(interval)
            
            else:
                # 未知状态，继续等待
                time.sleep(poll_interval_slow)
    
    def edit_image(
        self,
        function: str,
        prompt: str,
        base_image_url: str,
        mask_image_url: str = None,
        n: int = 1,
        seed: int = None,
        watermark: bool = False,
        strength: float = None,
        top_scale: float = None,
        bottom_scale: float = None,
        left_scale: float = None,
        right_scale: float = None,
        upscale_factor: int = None,
        is_sketch: bool = None,
        timeout: int = None,
    ) -> Dict[str, Any]:
        """
        完整的图像编辑流程：提交任务 -> 等待完成 -> 返回结果
        
        返回结果包含:
            - results: 结果图像URL列表
            - usage: 使用量信息
            - task_metrics: 任务统计信息
        """
        # 提交任务
        task_id = self.submit_task(
            function=function,
            prompt=prompt,
            base_image_url=base_image_url,
            mask_image_url=mask_image_url,
            n=n,
            seed=seed,
            watermark=watermark,
            strength=strength,
            top_scale=top_scale,
            bottom_scale=bottom_scale,
            left_scale=left_scale,
            right_scale=right_scale,
            upscale_factor=upscale_factor,
            is_sketch=is_sketch,
        )
        
        # 等待结果
        result = self.wait_for_result(task_id, timeout=timeout)
        
        return result
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = None,
        **kwargs
    ) -> requests.Response:
        """
        带重试的HTTP请求
        """
        max_retries = max_retries or self.MAX_RETRIES
        last_error = None
        
        # 合并headers
        if "headers" in kwargs:
            merged_headers = dict(self.session.headers)
            merged_headers.update(kwargs["headers"])
            kwargs["headers"] = merged_headers
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                
                # 检查HTTP状态码
                if response.status_code == 200:
                    return response
                
                # 处理错误响应
                try:
                    error_data = response.json()
                    error_code = error_data.get("code", f"HTTP_{response.status_code}")
                    error_msg = error_data.get("message", response.text)
                except:
                    error_code = f"HTTP_{response.status_code}"
                    error_msg = response.text
                
                # 某些错误不重试
                if response.status_code in [400, 401, 403, 404]:
                    raise WanxAPIError(code=error_code, message=error_msg)
                
                last_error = WanxAPIError(code=error_code, message=error_msg)
                
            except requests.exceptions.RequestException as e:
                last_error = WanxAPIError(
                    code="NETWORK_ERROR",
                    message=f"网络请求失败: {str(e)}"
                )
            
            # 重试前等待
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"[Wanx] 请求失败，{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        
        raise last_error


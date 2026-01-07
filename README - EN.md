# ComfyUI-HaoranWanxImageEdit

A ComfyUI plugin for Alibaba Cloud's **Tongyi Wanx 2.1 Image Editing** model.

## Features

This plugin wraps the `wanx2.1-imageedit` model from the Alibaba Cloud Model Studio (Bailian), supporting **10 distinct image editing functions**:

| Function | Description | Parameters |
| --- | --- | --- |
| **Global Stylization** | Transfer the entire image to a specified artistic style. | Edit Strength |
| **Local Stylization** | Apply style transfer only to a specific local area of the image. | - |
| **Instruction Editing** | Add or modify image content via natural language instructions. | Edit Strength |
| **Inpainting** | Edit specific areas using a mask. | Mask Required |
| **Watermark Removal** | Remove Chinese/English characters or watermarks from the image. | - |
| **Outpainting** | Expand the canvas in four directions with intelligent filling. | Expansion Ratios |
| **Super Resolution** | Enhance image clarity and support upscaling. | Scale Factor (1-4x) |
| **Colorization** | Convert black & white or grayscale images to color. | - |
| **Line Art to Image** | Generate a new image based on a line drawing/sketch. | Doodle Mode |
| **Cartoon Avatar** | Generate a new image based on a cartoon character. | - |

---

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes` directory.
2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Restart ComfyUI.

---

## API Key Configuration

### Method 1: Environment Variable (Recommended)

Set the environment variable `DASHSCOPE_API_KEY`:

* **Windows:**
```cmd
setx DASHSCOPE_API_KEY "sk-your-api-key"

```


* **Linux/Mac:**
```bash
export DASHSCOPE_API_KEY="sk-your-api-key"

```



### Method 2: Node Parameter

Directly input your API Key into the `api_key` field of the node.

### How to get an API Key

1. Visit the [Alibaba Cloud Model Studio Console](https://bailian.console.aliyun.com/?tab=model#/api-key).
2. Log in and create a new API Key.
3. New users typically receive a **500-image free quota**.

---

## Usage

### Basic Workflow

1. Search for the **"Haoran"** category in the node list.
2. Add the **"Haoran Tongyi Wanx Image Edit"** node.
3. Connect your input image.
4. Select the desired function and enter your prompt.
5. Execute the workflow.

### Node Parameters

#### Required Parameters

* **api_key**: Your API key (leave blank to use the environment variable).
* **image**: Input image.
* **function**: Select the editing mode.
* **prompt**: Describe the desired editing effect.

#### General Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| **n** | Number of images to generate (1-4). | 1 |
| **seed** | Random seed for reproducibility. | 0 |
| **use_seed** | Whether to enable the specific seed. | False |
| **ai_watermark** | Add an "AI Generated" watermark. | False |
| **timeout** | Maximum wait time in seconds. | 180 |
| **auto_resize** | Automatically scale image to valid range. | True |

#### Function-Specific Parameters

* **edit_strength**: (Global Stylization, Instruction Edit) 0.0-1.0; higher values mean more significant changes.
* **expand_ratio**: (Outpainting) Expansion ratio for Up/Down/Left/Right. 1.0 means no expansion.
* **upscale_factor**: (Super Resolution) Supports 1x to 4x.
* **sketch_mode**: (Line Art to Image) Toggle if the input is a hand-drawn doodle.

---

## Prompting Tips

* **Global Stylization**: Use "Convert to [Style]" format. E.g., `Convert to French picture book style`.
* **Local Stylization**: Use "Turn [Object] into [Style]" format. E.g., `Turn the house into wooden style`. Supported styles: Ice, Cloud, Lantern, Wood, Porcelain, Furry, Wool, Balloon.
* **Instruction Editing**: Use clear verbs like "Add" or "Change". E.g., `Give her a pair of sunglasses` or `Change hair color to red`.
* **Inpainting**: Describe the result for the mask area. E.g., `A puppy wearing a hat`. (Do not use "Delete XX").
* **Outpainting**: Describe the full desired scene after expansion. E.g., `A family having a picnic on a park lawn`.

---

## Image Requirements

* **Formats**: JPG, JPEG, PNG, BMP, TIFF, WEBP.
* **Resolution**: Width/Height between 512 and 4096 pixels.
* **File Size**: Maximum 10MB.
* *Tip: Enabling "auto_resize" will handle most compliance issues automatically.*

---

## Pricing & Quotas

* **Price**: 0.14 RMB / image.
* **Free Quota**: 500 images for new users.
* **Rate Limits**: RPS = 2, Max Concurrent Tasks = 2.

## Update Logs

### v1.0.0 (2026-01-07)

* Initial release.
* Full support for 10 image editing functions.
* Robust error handling and retry logic.

## License

[MIT License](https://www.google.com/search?q=LICENSE)

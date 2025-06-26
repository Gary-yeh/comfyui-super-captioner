# SuperCaptioner/captioner_node.py

import torch
from PIL import Image
import numpy as np
import comfy.model_management as model_management
import os

# --- 全域變數和輔助函式 ---
# 儲存已加載模型的字典，避免重複加載
loaded_models = {}

def get_blip_model(device, torch_dtype):
    """延遲加載 BLIP 模型"""
    if "blip" not in loaded_models:
        print("SuperCaptioner: Loading BLIP model...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = "Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(model_id,use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
        loaded_models["blip"] = {"model": model, "processor": processor}
        print("SuperCaptioner: BLIP model loaded.")
    return loaded_models["blip"]

def tensor_to_pil(image_tensor):
    """將 ComfyUI 的 Tensor 轉換為 PIL 影像列表"""
    pil_images = []
    for img in image_tensor:
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))
    return pil_images

# --- 核心節點類 ---
class SuperCaptionerNode:
    def __init__(self):
        self.device = model_management.get_torch_device()
        self.torch_dtype = torch.float16 if model_management.should_use_fp16() else torch.float32

    @classmethod
    def INPUT_TYPES(s):
        # 使用 COMBO 下拉選單讓使用者選擇模型
        return {
            "required": {
                "image": ("IMAGE",),
                "model_choice": (["blip-large", "gemini-1.5-flash"],),
            },
            "optional": {
                # Gemini 需要的額外輸入
                "google_api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "用繁體中文詳細描述這張圖片的內容。"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "execute"
    CATEGORY = "SuperCaptioner"

    def execute(self, image, model_choice, google_api_key="", prompt=""):
        captions = []
        pil_images = tensor_to_pil(image)

        if model_choice == "blip-large":
            # --- BLIP 處理邏輯 ---
            try:
                blip = get_blip_model(self.device, self.torch_dtype)
                model = blip["model"].to(self.device)
                processor = blip["processor"]
                
                inputs = processor(images=pil_images, return_tensors="pt").to(self.device, self.torch_dtype)
                generated_ids = model.generate(**inputs, max_new_tokens=75)
                captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # VRAM 管理：用完後立即釋放
                model.to("cpu")
                model_management.soft_empty_cache()

            except Exception as e:
                print(f"SuperCaptioner (BLIP) Error: {e}")
                return (f"BLIP Error: {e}",)

        elif model_choice == "gemini-pro-vision":
            # --- Gemini 處理邏輯 ---
            if not google_api_key or google_api_key.strip() == "":
                # 優先從環境變數讀取，增加安全性
                google_api_key = os.environ.get('GOOGLE_API_KEY', '')
                if not google_api_key:
                    return ("Error: Google API Key is required for Gemini. Please input it in the node or set GOOGLE_API_KEY environment variable.",)
            
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('gemini-pro-vision')
                
                for pil_image in pil_images:
                    response = model.generate_content([prompt, pil_image])
                    captions.append(response.text.strip())

            except Exception as e:
                print(f"SuperCaptioner (Gemini) Error: {e}")
                return (f"Gemini API Error: {e}",)
        
        final_caption = "\n".join(captions)
        print(f"SuperCaptioner Generated: {final_caption}")
        return (final_caption,)


# --- ComfyUI 註冊 ---
NODE_CLASS_MAPPINGS = {
    "SuperCaptioner": SuperCaptionerNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCaptioner": "Super Captioner"
}

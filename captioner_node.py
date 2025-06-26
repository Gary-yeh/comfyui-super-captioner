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
        # 修正 1: 移除 use_fast=True，它在新版 transformers 中已不推薦，且可能導致不一致
        # BlipProcessor 會自動處理好這些細節
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = "Salesforce/blip-image-captioning-large"
        
        # 將模型和處理器加載到 CPU，在使用時再轉移到 GPU，以更好地管理 VRAM
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
        
        loaded_models["blip"] = {"model": model, "processor": processor}
        print("SuperCaptioner: BLIP model loaded.")
    return loaded_models["blip"]

def tensor_to_pil(image_tensor):
    """將 ComfyUI 的 Tensor 轉換為 PIL 影像列表"""
    pil_images = []
    for img in image_tensor:
        # 確保張量在 CPU 上才能轉換為 numpy
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
        # 修正 2: 讓下拉選單的選項名稱更具描述性且唯一，避免混淆
        # 我們將使用這些字串作為清晰的標識符
        s.MODEL_CHOICES = [
            "local: blip-large", 
            "google: gemini-2.0-flash",
            "google: gemini-2.0-flash-lite" # 增加一個選項作為範例
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "model_choice": (s.MODEL_CHOICES,),
            },
            "optional": {
                "google_api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "用繁體中文詳細描述這張圖片的內容。"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "execute"
    CATEGORY = "SuperCaptioner"

    def execute(self, image, model_choice, google_api_key="", prompt=""):
        # 每次執行都從一個空的 caption 列表開始，這可以確保清空舊內容
        captions = []
        pil_images = tensor_to_pil(image)

        try:
            # 修正 3: 使用更清晰的邏輯判斷來選擇模型分支
            if model_choice.startswith("local:"):
                captions = self.execute_blip(pil_images)
            elif model_choice.startswith("google:"):
                # 從選項字串中提取出真正的模型名稱
                model_name = model_choice.split("google: ")[1]
                captions = self.execute_gemini(pil_images, model_name, google_api_key, prompt)
            else:
                return (f"Error: Unknown model choice '{model_choice}'",)
        
        except Exception as e:
            # 增加一個頂層的異常捕獲，防止整個節點崩潰
            error_message = f"An unexpected error occurred: {e}"
            print(f"SuperCaptioner Error: {error_message}")
            return (error_message,)

        # 修正 4: 在返回前，確保最終結果是一個格式正確的字串
        final_caption = "\n".join(captions).strip()
        print(f"SuperCaptioner Generated: {final_caption}")
        
        # ComfyUI 的輸出需要是一個包含元組的元組
        return (final_caption,)

    def execute_blip(self, pil_images):
        """執行 BLIP 模型的獨立函數，使程式碼更整潔"""
        blip = get_blip_model(self.device, self.torch_dtype)
        model = blip["model"].to(self.device)
        processor = blip["processor"]
        
        inputs = processor(images=pil_images, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=75)
        blip_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # VRAM 管理：用完後立即釋放
        model.to("cpu")
        model_management.soft_empty_cache()
        
        return blip_captions

    def execute_gemini(self, pil_images, model_name, api_key, prompt):
        """執行 Gemini 模型的獨立函數"""
        if not api_key or api_key.strip() == "":
            api_key = os.environ.get('GOOGLE_API_KEY', '')
            if not api_key:
                # 修正 5: 拋出異常而不是直接返回，讓頂層 try-except 捕獲
                raise ValueError("Google API Key is required. Please input it or set GOOGLE_API_KEY env var.")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # 使用傳入的模型名稱
        model = genai.GenerativeModel(model_name)
        
        gemini_captions = []
        for pil_image in pil_images:
            response = model.generate_content([prompt, pil_image])
            gemini_captions.append(response.text.strip())
        
        return gemini_captions

# --- ComfyUI 註冊 ---
NODE_CLASS_MAPPINGS = {
    "SuperCaptioner": SuperCaptionerNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCaptioner": "Super Captioner"
}

# SuperCaptioner/captioner_node.py

import torch
from PIL import Image
import numpy as np
import comfy.model_management as model_management
import os
import logging # 1. 導入 logging 模組

# 2. 設定一個專屬於此模組的 logger
#    使用 __name__ 可以讓日誌記錄器自動獲得 "captioner_node" 這個名字，方便追蹤。
logger = logging.getLogger(__name__)

# --- 全域變數和輔助函式 ---
loaded_models = {}

def get_blip_model(device, torch_dtype):
    """延遲加載 BLIP 模型"""
    if "blip" not in loaded_models:
        # 將 print 改為 logger.info
        logger.info("SuperCaptioner: Loading BLIP model...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = "Salesforce/blip-image-captioning-large"
        
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
        
        loaded_models["blip"] = {"model": model, "processor": processor}
        logger.info("SuperCaptioner: BLIP model loaded.")
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
        # 修正：使用更標準的 Gemini 模型名稱
        s.MODEL_CHOICES = [
            "local: blip-large", 
            "google: gemini-2.5-flash",
            "google: gemini-2.0-flash",
            "google: gemini-2.0-flash-lite"
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
        captions = []
        pil_images = tensor_to_pil(image)

        try:
            if model_choice.startswith("local:"):
                captions = self.execute_blip(pil_images)
            elif model_choice.startswith("google:"):
                model_name = model_choice.split("google: ")[1].strip()
                captions = self.execute_gemini(pil_images, model_name, google_api_key, prompt)
            else:
                return (f"Error: Unknown model choice '{model_choice}'",)
        
        except Exception as e:
            # 將 print 改為 logger.error，用於記錄嚴重錯誤
            # exc_info=True 會自動記錄完整的錯誤堆疊追蹤，對除錯非常有幫助
            logger.error(f"SuperCaptioner Error: An unexpected error occurred: {e}", exc_info=True)
            return (f"An unexpected error occurred: {e}",)

        final_caption = "\n".join(captions).strip()
        
        # 3. 移除最後的 print 語句
        # 替換為 DEBUG 等級的日誌，預設不會顯示，但可以在開發時啟用
        logger.debug(f"SuperCaptioner Generated: {final_caption}")
        
        return (final_caption,)

    def execute_blip(self, pil_images):
        """執行 BLIP 模型的獨立函數，使程式碼更整潔"""
        blip = get_blip_model(self.device, self.torch_dtype)
        model = blip["model"].to(self.device)
        processor = blip["processor"]
        
        inputs = processor(images=pil_images, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=75)
        blip_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        model.to("cpu")
        model_management.soft_empty_cache()
        
        return blip_captions

    def execute_gemini(self, pil_images, model_name, api_key, prompt):
        """執行 Gemini 模型的獨立函數"""
        if not api_key or api_key.strip() == "":
            api_key = os.environ.get('GOOGLE_API_KEY', '')
            if not api_key:
                raise ValueError("Google API Key is required. Please input it or set GOOGLE_API_KEY env var.")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
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

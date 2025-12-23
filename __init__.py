import os
import sys

import gc
import torch
import librosa
import torchaudio

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(now_dir, 'third_party/Matcha-TTS'))
sys.path.append(now_dir)
model_dir = os.path.join(now_dir, "pretrained_models")
if not os.path.exists(os.path.join(model_dir,"CosyVoice2-0.5B")):
    print("download.......CosyVoice")
    from modelscope import snapshot_download
    snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/CosyVoice3-0.5B')
    snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
    os.system(f'cd {model_dir}/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && apt install -y unzip && unzip resource.zip -d .')

from cosyvoice.cli.cosyvoice import  AutoModel
from cosyvoice.utils.common import set_all_random_seed

max_val = 0.8
 
class Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (os.listdir(model_dir),),
                "load_trt": ("BOOLEAN", {"default": False},),
                "load_jit": ("BOOLEAN", {"default": False},),
                "trt_concurrent":("INT",{"default": 1}),
            },
        }
    RETURN_TYPES = ("MODEL_CosyVoice",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"
    def run(self, model,load_trt, load_jit, trt_concurrent):
        print("GKK·CosVoice: Model loading")
        return (AutoModel(model_dir=os.path.join(model_dir,model), load_jit=load_jit, load_trt=load_trt, trt_concurrent=trt_concurrent),)
 

class CosyVoice():
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "text": ("TEXT",),
                "mode": (['3s复刻', '跨语种复刻', '语言控制'],{
                    "default": "3s复刻"
                }),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
            "optional":{
                "prompt": ("TEXT", ),
                "instuct": ("TEXT", ),
            }
        }

    RETURN_TYPES = ("AUDIO","speechs_dict")
    RETURN_NAMES = ("audio","speechs")
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"
    __model = None

    def run(self, model, audio, text, mode, speed, seed,  prompt=None, instuct=None, concat= False):
        set_all_random_seed(seed)
        speechs = []
        print("GKK·CosVoice: Start infer")
        if mode == "跨语种复刻": 
            speechs = [i["tts_speech"] for i in model.inference_cross_lingual(text, audio,  speed=speed)]
        elif mode == "3s复刻":
            assert prompt is not None , '3s极速复刻 need prompt input'
            speechs = [i["tts_speech"] for i in model.inference_zero_shot(text, "You are a helpful assistant.<|endofprompt|>"+prompt, audio, speed=speed)]
        elif mode == "语言控制":
            assert instuct is not None , '自然语言控制 need instuct input'
            speechs = [i["tts_speech"] for i in model.inference_instruct2(text, "You are a helpful assistant."+instuct+"<|endofprompt|>", audio,  speed=speed)]
        gc.collect()
        torch.cuda.empty_cache()
        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        if concat:
            print("GKK·CosVoice: return speechs")
            return  ({"waveform": tts_speech, "sample_rate": model.sample_rate}, {"speechs":speechs, "sample_rate": model.sample_rate},)
        else:
            return ( {"waveform": tts_speech, "sample_rate": model.sample_rate},)

class Copy3s(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "prompt": ("TEXT", ),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,model,audio,prompt, text, speed, seed, concat=False):
        return super().run(model,audio,text,"3s复刻",speed,seed,prompt,concat=concat)

class CrossLingual(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,model,audio, text, speed, seed):
        return super().run(model,audio,text,"跨语种复刻",speed,seed)

class NLControl(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "instuct": ("TEXT", ),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,model,audio,instuct, text, speed, seed):
        return super().run(model,audio,text,"语言控制",speed,seed,instuct=instuct)
    
class Input:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "prompt": ("STRING",{ "dynamicPrompts": True,}),
                        "text": ("STRING", {
                            "multiline": True,
                            "dynamicPrompts": True,
                            "style": "resize: vertical;",
                            "oninput": "this.style.height = 'auto'; this.style.height = (this.scrollHeight) + 'px';" 
                        })
                    }
                }
    RETURN_TYPES = ("TEXT","TEXT")
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"
    def run(self,prompt,text):
        return (prompt,text)


NODE_CLASS_MAPPINGS = {
    "Text2":Input,
    "CosyVoiceLoader":Loader,
    "CosyVoice3s":Copy3s,
    "CosyVoiceNLControl":NLControl,
    "CosyVoiceCrossLingual":CrossLingual,
}

__all__ = ['NODE_CLASS_MAPPINGS']
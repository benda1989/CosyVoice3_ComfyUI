import os
import sys
import torchaudio
import gradio as gr
from funasr import AutoModel

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.common import set_all_random_seed

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(now_dir, 'third_party/Matcha-TTS'))
sys.path.append(now_dir)
model_dir = os.path.join(now_dir, "pretrained_models")

if not os.path.exists(os.path.join(model_dir, "CosyVoice3-0.5B")):
    print("download.......CosyVoice")
    from modelscope import snapshot_download
    snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/CosyVoice3-0.5B')
    snapshot_download('iic/CosyVoice-ttsfrd',
                      local_dir='pretrained_models/CosyVoice-ttsfrd')
    os.system(f'cd {model_dir}/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && apt install -y unzip && unzip resource.zip -d .')

max_val = 0.8

class CosVoice(CosyVoice3):
    _load = False
    _data = []

    def getData(self, index=None):
        if index is not None:
            yield self._data[index]
        else:
            for i in self._data:
                yield i

    def exec(self, *arg, **kwarg):
        yield from self._model(*arg, **kwarg)

    def getTTS(self, mode="自然语言控制", prompt_text=None, prompt_speech=None, instruct_text=None, spk_id=""):
        self.prompt_speech = prompt_speech
        self.prompt_text = self.frontend.text_normalize(
            prompt_text, split=False) if prompt_text else None
        self.instruct_text = self.frontend.text_normalize(
            "You are a helpful assistant."+instruct_text+"<|endofprompt|>", split=False) if instruct_text else None
        self.spk_id = spk_id
        self._load = True
        self._data = []
        self._model = None

        if mode == '预训练音色':
            assert self.spk_id, '没有可用的预训练音色！'
            self._model = self.sft
        elif mode == '3s极速复刻':
            assert  prompt_text ,'prompt文本为空'
            assert  prompt_speech is not None, 'prompt音频为空'
            self._model = self.zero_shot
        elif mode == '跨语种复刻':
            assert  prompt_speech is not None, 'prompt音频为空'
            self._model = self.cross_lingual
        else:
            assert instruct_text, 'instruct文本为空'
            self._model = self.instruct2
        return self._model

    def instructText(self, text=None):
        if text:
            return self.frontend.text_normalize("You are a helpful assistant."+text+"<|endofprompt|>", split=False)
        else:
            return self.instruct_text

    def sft(self, text, stream=False, speed=1.0, index=None):
        ps = self.frontend.frontend_sft(text, self.spk_id)
        yield from self.tts(ps, stream, speed, index)

    def zero_shot(self, text, stream=False, speed=1.0, index=None):
        ps = self.frontend.frontend_zero_shot(text, "You are a helpful assistant.<|endofprompt|>"+self.prompt_text, self.prompt_speech, self.sample_rate)
        yield from self.tts(ps, stream, speed, index)

    def cross_lingual(self, text, stream=False, speed=1.0, index=None):
        ps = self.frontend.frontend_cross_lingual(text, self.prompt_speech, self.sample_rate)
        yield from self.tts(ps, stream, speed, index)

    def instruct(self, text, stream=False, speed=1.0, instruct_text=None, index=None):
        ps = self.frontend.frontend_instruct(text, self.spk_id, self.instructText(instruct_text))
        yield from self.tts(ps, stream, speed, index)

    def instruct2(self, text, stream=False, speed=1.0, instruct_text=None, index=None):
        ps = self.frontend.frontend_instruct2(text,  self.instructText(instruct_text), self.prompt_speech, self.sample_rate)
        yield from self.tts(ps, stream, speed, index)

    def tts(self, model_input, stream, speed, index=None):
        for i in self.model.tts(**model_input, stream=stream, speed=speed):
            if index is not None:
                self._data[index] = i['tts_speech']
            else:
                self._data.append(i['tts_speech'])
            yield  i['tts_speech'] 

    def split_text(self, text):
        self._data = []
        return "\n".join(self.frontend.text_normalize(text, split=True))

    def savePath(self, path):
        for i in self._data:
            torchaudio.save(os.path.join(path, 'sft_{}.wav'.format(i+300)),
                            i,
                            self.sample_rate)


def generate_seed():
    import random
    return {
        "__type__": "update",
        "value": random.randint(1, 100000000)
    }


def prompt_wav_recognition(prompt_wav,mode):
    if mode == '3s极速复刻':
        res = asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        return  res[0]["text"].split('|>')[-1]
    return ""


def generate(tts_text, mode_box, prompt_text, prompt_wav, instruct_text, seed, stream, speed, status_list):
    set_all_random_seed(seed)
    lines = tts_text.splitlines()
    model = cosyvoice.getTTS(mode_box, prompt_text, prompt_wav , instruct_text)
    status = []
    for idx, text in enumerate(lines):
        status.append({
            "text": text,
            "status": "waiting",
            "index": idx
        })
    
    yield None, status
    
    for idx, text in enumerate(lines):
        print(f"生成第 {idx+1}/{len(lines)} 句: {text}")
        
        status[idx]["status"] = "generating"
        yield None, status
        
        for i in model(text, stream=stream, speed=speed):
            audio_data = (cosyvoice.sample_rate, i.numpy().flatten())
        
        status[idx]["status"] = "completed"
        yield audio_data, status


def final():
    for i in cosyvoice.getData():
        yield (cosyvoice.sample_rate,i.numpy().flatten())
def single():
    for i, j in enumerate(cosyvoice.getData()):
        torchaudio.save('vc_{}.wav'.format(i), j, cosyvoice.sample_rate)

def main():
    instruct_dict = {'3s极速复刻': '1. 选择音频文件\n2. 修改prompt文本\n3. 输入合成文本\n4. (可选)点击自动分割成句\n5. 点击生成音频',
                     '自然语言控制': '1. 选择音频文件\n2. 输入instruct文本\n3. 输入合成文本\n4. (可选)点击自动分割成句\n5. 点击生成音频',
                     "跨语种复刻": "1. 选择音频文件\n2. 输入合成文本\n3. (可选)点击自动分割成句\n4. 点击生成音频"
                    }
    inference_mode_list = [i for i in instruct_dict.keys()]

    with gr.Blocks() as demo:
        with gr.Row():
            mode_box = gr.Radio(choices=inference_mode_list,
                                label='选择模式',
                                value=inference_mode_list[0])
            instruction_text = gr.Text(
                label="操作步骤",
                value=instruct_dict[inference_mode_list[0]])
            stream = gr.Radio(
                choices=[('否', False), ('是', True)],
                label='流式推理',value=False,scale=0.5)
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=36051420, label="随机推理种子")
        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz，控制30s以内')
            with gr.Column():
                prompt_text = gr.Textbox(
                    label="prompt文本", 
                    placeholder="请输入prompt文本，支持自动识别，您可以自行修正识别结果...", 
                    value='',
                    visible=True
                )
                instruct_text = gr.Textbox(
                    label="输入instruct文本", 
                    placeholder="请输入instruct文本.例如:用四川话说这句话。", 
                    value='',
                    visible=False
                )
        tts_text = gr.Textbox(label="输入合成文本，注意换行(shift+enter)切分句子，单行不可过长，", value="""CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。
CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.""")

        with gr.Row():
            with gr.Column(scale=0.1):
                split_button = gr.Button("自动分割成句", variant="secondary", interactive=False)
                generate_button = gr.Button("生成音频", variant="primary", interactive=False)
            audio_output = gr.Audio(label="结果展示，生成结束点击右侧小图标下载", autoplay=True, streaming=True)
        with gr.Row():
            final_btn = gr.Button("重新合成", visible=False, variant="primary")
            single_btn = gr.Button("按行保存", visible=False,)
        with gr.Column(visible=False) as edit_container:
            gr.Markdown("### 🎵 句子编辑与音频重新生成，修改结束执行‘重新合成’按钮")
        
        sentences_state = gr.State([])
        status_list_state = gr.State([])
        
        @gr.render(inputs=[status_list_state, mode_box, instruct_text], triggers=[status_list_state.change])
        def render_edit_interface(status_list, mode_box, instruct_text_value):
            if not status_list:
                return
            
            with gr.Column():
                for item in status_list:
                    idx = item["index"]
                    text = item["text"]
                    status = item["status"]
                    
                    with gr.Row():
                        with gr.Column(scale=4):
                            texts = gr.Textbox(
                                text, 
                                interactive=True, 
                                label=f"句子 {idx+1}",
                                show_label=True
                            )
                            
                            if status == "completed":
                                index_component = gr.Number(idx, visible=False)
                                with gr.Row():
                                    if mode_box == "自然语言控制":
                                        instruct_component = gr.Textbox(
                                            show_label=False, 
                                            placeholder="instruct(可选)", 
                                            value=instruct_text_value,
                                            container=False,
                                            scale=2
                                        )
                                        regen_btn = gr.Button("重新生成",  variant="secondary", scale=1)
                                    else:
                                        regen_btn = gr.Button("重新生成",  variant="secondary")
                        
                        if status == "completed" and idx < len(cosyvoice._data) and cosyvoice._data:
                            with gr.Column(scale=3):
                                try:
                                    audio_component = gr.Audio(
                                        (cosyvoice.sample_rate, cosyvoice._data[idx].numpy().flatten()),
                                        scale=1,
                                        show_label=False,
                                        interactive=False
                                    )
                                except:
                                    audio_component = gr.Audio(
                                        None, 
                                        scale=1, 
                                        show_label=False,
                                        visible=False
                                    )
                                    
                                if status == "completed":
                                    if mode_box == "自然语言控制":
                                        regen_btn.click(
                                            lambda x,y,z: (cosyvoice.sample_rate, next(cosyvoice.exec(text=x, instruct_text=z, index=y)).numpy().flatten()),
                                            [texts, index_component, instruct_component],
                                            audio_component
                                        )
                                    else:
                                        regen_btn.click(
                                            lambda x,y: (cosyvoice.sample_rate, next(cosyvoice.exec(text=x, index=y)).numpy().flatten()),
                                            [texts, index_component],
                                            audio_component
                                        )
                        else:
                            with gr.Column(scale=1):
                                if status == "generating":
                                    gr.Markdown("*生成中...*")
                                elif status == "waiting":
                                    gr.Markdown("*等待中*")
                                else:
                                    gr.Markdown("")
        seed_button.click(generate_seed, outputs=seed)
        def update_interface_by_mode(mode):
            if mode == '3s极速复刻':
                return (
                    instruct_dict[mode],
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
            elif mode == '自然语言控制':
                return (
                    instruct_dict[mode],
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            elif mode == '跨语种复刻':
                return (
                    instruct_dict[mode],
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            else:
                return (
                    instruct_dict.get(mode, ""),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        mode_box.change(update_interface_by_mode,
                        mode_box,
                        [instruction_text, prompt_text, instruct_text])
        def check_prerequisites(mode, prompt_wav, prompt_text_val, instruct_text_val, tts_text_val):
            has_tts_text = bool(tts_text_val and tts_text_val.strip())
            
            if mode == '3s极速复刻':
                ready = bool(prompt_wav and prompt_text_val and has_tts_text)
            elif mode == '自然语言控制':
                ready = bool(prompt_wav and instruct_text_val and has_tts_text)
            elif mode == '跨语种复刻':
                ready = bool(prompt_wav and has_tts_text)
            else:
                ready = has_tts_text
            
            split_enabled = has_tts_text
            generate_enabled = ready
            
            return (
                gr.update(interactive=split_enabled),
                gr.update(interactive=generate_enabled)
            )
        
        input_components = [mode_box, prompt_wav_upload, prompt_text, instruct_text, tts_text]
        output_components = [split_button, generate_button]
        
        for component in input_components:
            component.change(
                check_prerequisites,
                inputs=input_components,
                outputs=output_components
            )
        
        prompt_wav_upload.change(prompt_wav_recognition,
                                 [prompt_wav_upload,mode_box],
                                 prompt_text)
        split_button.click(cosyvoice.split_text,
                           tts_text,
                           tts_text)
        def prepare_and_generate(tts_text):
            lines = tts_text.splitlines()
            max_line_length = 1000
            for idx, line in enumerate(lines):
                if len(line) > max_line_length:
                    gr.Warning(f"第{idx+1}句超过{max_line_length}字符，请使用'自动分割成句'功能")
                    return gr.update(visible=False), [], []
            
            initial_status = []
            for idx, text in enumerate(lines):
                initial_status.append({
                    "text": text,
                    "status": "waiting",
                    "index": idx
                })
            
            return gr.update(visible=True), lines, initial_status
        
        generate_button.click(
            prepare_and_generate,
            inputs=tts_text,
            outputs=[edit_container, sentences_state, status_list_state]
        ).then(
            generate,
            inputs=[tts_text, mode_box, prompt_text, prompt_wav_upload, instruct_text, seed, stream, speed, status_list_state],
            outputs=[audio_output, status_list_state]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[final_btn, single_btn]
        )
        final_btn.click(final,
                        outputs=audio_output)
        single_btn.click(single)    
    demo.launch( server_name='0.0.0.0')


if __name__ == '__main__':
    cosyvoice = CosVoice('pretrained_models/CosyVoice3-0.5B')
    asr_model = AutoModel(
        model="iic/SenseVoiceSmall",
        disable_update=True,
        log_level='DEBUG',
        device="cuda:0")
    main()

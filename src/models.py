import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, SeamlessM4TModel, SeamlessM4Tv2Model, AutoProcessor, pipeline
from seamless_communication.models.unity import UnitYModel
from seamless_communication.inference import Translator
from typing import Tuple, Iterable, Dict, Any
from scipy.io.wavfile import write

class Whisper:
    def __init__(self, model):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model, torch_dtype=torch_dtype, use_safetensors=True)
        self.model.to(device)
        
        self.processor = AutoProcessor.from_pretrained(model)
        
        self. pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            device=device
        )
    
    def forward(self, x):
        if type(x) == str:
            x, orig_freq = torchaudio.load(x)
            x = torchaudio.functional.resample(x, orig_freq=orig_freq, new_freq=16_000).squeeze().numpy()
        transcription = self.pipe(x, generate_kwargs={"language": "urdu"})
        return transcription["text"]

class MMS:
    def __init__(self, model, finetuned=False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        TARGET_LANG = "urd-script_arabic"
        if finetuned:
            self.pipe = pipeline(
                model=model,
                model_kwargs={"ignore_mismatched_sizes": True},
                torch_dtype=torch_dtype,
                device=device
            )
        else:
            self.pipe = pipeline(
                model=model,
                model_kwargs={"target_lang": TARGET_LANG, "ignore_mismatched_sizes": True},
                torch_dtype=torch_dtype,
                device=device
            )
    
    def forward(self, x):
        if type(x) == str:
            x, orig_freq = torchaudio.load(x)
            x = torchaudio.functional.resample(x, orig_freq=orig_freq, new_freq=16_000).squeeze().numpy()
        transcription = self.pipe(x)
        return transcription["text"]

class SeamlessM4T:
    def __init__(self, model):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(model)
        if "v2" in model:
            self.model = SeamlessM4Tv2Model.from_pretrained(model).to(self.device)
        else:
            self.model = SeamlessM4TModel.from_pretrained(model).to(self.device)
    
    def forward(self, x):
        TGT_LANG = "urd"
        if type(x) == str:
            x, orig_freq = torchaudio.load(x)
            x = torchaudio.functional.resample(x, orig_freq=orig_freq, new_freq=16_000)
        audio_inputs = self.processor(audios=x, sampling_rate=16000, return_tensors="pt")
        output_tokens = self.model.generate(**audio_inputs.to(self.device), tgt_lang=TGT_LANG, generate_speech=False)
        text_from_audio = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return text_from_audio

class SeamlessM4TFinetuned:
    def __init__(self, model):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if "large" in model:
            self.translator = Translator(
                model_name_or_card="seamlessM4T_v2_large",
                vocoder_name_or_card=None,
                device=torch.device("cuda")
            )

            self.load_checkpoint(self.translator.model, "seamless-m4t/v2-large/checkpoint.pt", torch.device(self.device))
        elif "medium" in model:
            self.translator = Translator(
                model_name_or_card="seamlessM4T_medium",
                vocoder_name_or_card=None,
                device=torch.device("cuda")
            )

            self.load_checkpoint(self.translator.model, "seamless-m4t/medium/checkpoint.pt", torch.device(self.device))
    
    def load_checkpoint(self, model: UnitYModel, path: str, device = "cpu") -> None:
        state_dict = torch.load(path, map_location=device)["model"]

        def _select_keys(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
            return {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}

        model.speech_encoder_frontend.load_state_dict(_select_keys(state_dict, "model.speech_encoder_frontend."))
        model.speech_encoder.load_state_dict(_select_keys(state_dict, "model.speech_encoder."))

        assert model.text_decoder_frontend is not None
        model.text_decoder_frontend.load_state_dict(_select_keys(state_dict, "model.text_decoder_frontend."))

        assert model.text_decoder is not None
        model.text_decoder.load_state_dict(_select_keys(state_dict, "model.text_decoder."))
    
    def forward(self, x):
        TGT_LANG = "urd"
        if type(x) == str:
            x, orig_freq = torchaudio.load(x)
            x = torchaudio.functional.resample(x, orig_freq=orig_freq, new_freq=16_000)
            prediction = self.translator.predict(input=x.T, task_str="s2tt", tgt_lang="urd", src_lang="urd")[0][0]
        else:
            sample_rate = 16000
            write('output.wav', sample_rate, x)
            prediction = self.translator.predict(input='output.wav', task_str="s2tt", tgt_lang="urd", src_lang="urd")[0][0]
        prediction = str(prediction)

        return prediction


"""
Applio ComfyUI Custom Nodes
Fork of IAHispano/Applio — ComfyUI 통합 레이어

노드 목록:
  ApplioModelLoader    : .pth + .index 모델 로드
  ApplioInfer          : 음성 변환 (전체 파라미터)
  ApplioReverb         : 리버브
  ApplioChorus         : 코러스
  ApplioDelay          : 딜레이
  ApplioCompressor     : 컴프레서
  ApplioGain           : 게인
  ApplioLimiter        : 리미터
  ApplioPitchShiftFX   : 피치 시프트 (후처리용)
  ApplioDistortion     : 디스토션
  ApplioBitcrush       : 비트크러시
  ApplioClipping       : 클리핑
  ApplioVoiceBlender   : 모델 블렌딩
  ApplioTTS            : EdgeTTS 텍스트-음성 변환
  ApplioAudioAnalyzer  : 스펙트럼 분석
"""

import os
import sys
import asyncio
import tempfile
import traceback

import torch
import numpy as np
import soundfile as sf

from rvc.infer.infer import VoiceConverter
from rvc.lib.tools.analyzer import analyze_audio
from rvc.train.process.model_blender import model_blender
from pedalboard import (
    Pedalboard, Reverb, Chorus, Delay, Compressor,
    Gain, Limiter, PitchShift, Distortion, Bitcrush, Clipping,
)
import edge_tts

# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

_VC: VoiceConverter | None = None


def _get_vc() -> VoiceConverter:
    global _VC
    if _VC is None:
        _VC = VoiceConverter()
    return _VC


def _to_wav(audio_dict: dict) -> str:
    """ComfyUI AUDIO → 임시 WAV 경로"""
    waveform = audio_dict["waveform"]       # [B, C, T]
    sr = audio_dict["sample_rate"]
    arr = waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, arr, sr)
    tmp.close()
    return tmp.name


def _from_wav(path: str) -> dict:
    """WAV 경로 → ComfyUI AUDIO"""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": sr}


def _apply_board(audio_dict: dict, board: Pedalboard) -> dict:
    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]
    arr = waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)
    out = board(arr, sr)
    tensor = torch.from_numpy(out.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": sr}


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────
# 1. ApplioModelLoader
# ─────────────────────────────────────────────────────────────

class ApplioModelLoader:
    """학습된 화자 모델(.pth)과 인덱스 파일(.index)을 로드합니다."""

    CATEGORY = "Audio/Applio"
    RETURN_TYPES  = ("APPLIO_MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pth_path": ("STRING", {
                "default": "",
                "tooltip": "학습된 화자 .pth 체크포인트 경로"
            }),
            "index_path": ("STRING", {
                "default": "",
                "tooltip": ".index 파일 경로 (없으면 빈 문자열)"
            }),
            "embedder_model": ([
                "contentvec", "spin", "spin-v2",
                "chinese-hubert-base", "japanese-hubert-base",
                "korean-hubert-base", "custom",
            ], {"default": "contentvec"}),
            "embedder_custom_path": ("STRING", {
                "default": "",
                "tooltip": "embedder_model=custom 일 때 경로"
            }),
        }}

    def load(self, pth_path, index_path, embedder_model, embedder_custom_path):
        return ({"pth_path": pth_path,
                 "index_path": index_path,
                 "embedder_model": embedder_model,
                 "embedder_custom_path": embedder_custom_path or None},)


# ─────────────────────────────────────────────────────────────
# 2. ApplioInfer
# ─────────────────────────────────────────────────────────────

class ApplioInfer:
    """Applio 음성 변환. 모든 추론 파라미터를 지원합니다."""

    CATEGORY = "Audio/Applio"
    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "infer"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":   ("APPLIO_MODEL",),
            "audio":   ("AUDIO",),
            "pitch":   ("INT",   {"default": 0,    "min": -24,  "max": 24,
                                   "tooltip": "반음 단위 피치 이동"}),
            "f0_method": (["rmvpe", "crepe"], {"default": "rmvpe"}),
            "index_rate": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "FAISS 인덱스 블렌드 (0=무시, 1=완전 적용)"}),
            "volume_envelope": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            "protect": ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 0.5,  "step": 0.01,
                                   "tooltip": "무성 자음 보호 (0.5=최대)"}),
            "hop_length": ("INT", {"default": 128, "min": 32,  "max": 512,  "step": 32}),
            "split_audio":         ("BOOLEAN", {"default": False}),
            "clean_audio":         ("BOOLEAN", {"default": False}),
            "clean_strength":      ("FLOAT",   {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            "f0_autotune":         ("BOOLEAN", {"default": False}),
            "f0_autotune_strength":("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            "formant_shifting":    ("BOOLEAN", {"default": False,
                                                "tooltip": "포르만트 이동 활성화"}),
            "formant_qfrency":     ("FLOAT",   {"default": 0.8, "min": 0.0, "max": 16.0, "step": 0.1}),
            "formant_timbre":      ("FLOAT",   {"default": 0.8, "min": 0.0, "max": 16.0, "step": 0.1}),
            "sid": ("INT", {"default": 0, "min": 0, "max": 100,
                            "tooltip": "멀티-스피커 모델의 화자 ID"}),
        }}

    def infer(self, model, audio, pitch, f0_method, index_rate, volume_envelope,
              protect, hop_length, split_audio, clean_audio, clean_strength,
              f0_autotune, f0_autotune_strength, formant_shifting,
              formant_qfrency, formant_timbre, sid):

        vc = _get_vc()
        in_path  = _to_wav(audio)
        out_tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = out_tmp.name
        out_tmp.close()

        try:
            vc.convert_audio(
                audio_input_path=in_path,
                audio_output_path=out_path,
                model_path=model["pth_path"],
                index_path=model["index_path"],
                pitch=pitch,
                f0_method=f0_method,
                index_rate=index_rate,
                volume_envelope=volume_envelope,
                protect=protect,
                hop_length=hop_length,
                split_audio=split_audio,
                f0_autotune=f0_autotune,
                f0_autotune_strength=f0_autotune_strength,
                embedder_model=model["embedder_model"],
                embedder_model_custom=model["embedder_custom_path"],
                clean_audio=clean_audio,
                clean_strength=clean_strength,
                export_format="WAV",
                post_process=False,
                sid=sid,
                formant_shifting=formant_shifting,
                formant_qfrency=formant_qfrency,
                formant_timbre=formant_timbre,
            )
            result = _from_wav(out_path)
        except Exception as e:
            print(f"[Applio] Infer error: {e}\n{traceback.format_exc()}")
            result = audio
        finally:
            for p in (in_path, out_path):
                try: os.remove(p)
                except OSError: pass

        return (result,)


# ─────────────────────────────────────────────────────────────
# 3~12. 이펙트 노드 (체이닝 가능)
# ─────────────────────────────────────────────────────────────

class ApplioReverb:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":       ("AUDIO",),
            "room_size":   ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,  "step": 0.05}),
            "damping":     ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,  "step": 0.05}),
            "wet_level":   ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0,  "step": 0.05}),
            "dry_level":   ("FLOAT", {"default": 0.4,  "min": 0.0, "max": 1.0,  "step": 0.05}),
            "width":       ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0,  "step": 0.05}),
            "freeze_mode": ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 1.0,  "step": 0.05}),
        }}
    def apply(self, audio, room_size, damping, wet_level, dry_level, width, freeze_mode):
        return (_apply_board(audio, Pedalboard([Reverb(
            room_size=room_size, damping=damping, wet_level=wet_level,
            dry_level=dry_level, width=width, freeze_mode=freeze_mode)])),)


class ApplioChorus:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":         ("AUDIO",),
            "rate_hz":       ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0, "step": 0.1}),
            "depth":         ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,  "step": 0.05}),
            "center_delay":  ("FLOAT", {"default": 7.0,  "min": 0.0, "max": 100.0,"step": 1.0}),
            "feedback":      ("FLOAT", {"default": 0.0,  "min":-1.0, "max": 1.0,  "step": 0.05}),
            "mix":           ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,  "step": 0.05}),
        }}
    def apply(self, audio, rate_hz, depth, center_delay, feedback, mix):
        return (_apply_board(audio, Pedalboard([Chorus(
            rate_hz=rate_hz, depth=depth, centre_delay_ms=center_delay,
            feedback=feedback, mix=mix)])),)


class ApplioDelay:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":    ("AUDIO",),
            "delay_s":  ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0,  "step": 0.05}),
            "feedback": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,  "step": 0.05}),
            "mix":      ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,  "step": 0.05}),
        }}
    def apply(self, audio, delay_s, feedback, mix):
        return (_apply_board(audio, Pedalboard([Delay(
            delay_seconds=delay_s, feedback=feedback, mix=mix)])),)


class ApplioCompressor:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":      ("AUDIO",),
            "threshold":  ("FLOAT", {"default":-20.0, "min":-60.0, "max": 0.0,   "step": 1.0}),
            "ratio":      ("FLOAT", {"default":  4.0, "min":  1.0, "max":20.0,   "step": 0.5}),
            "attack_ms":  ("FLOAT", {"default": 10.0, "min":  0.1, "max":500.0,  "step": 1.0}),
            "release_ms": ("FLOAT", {"default":100.0, "min": 10.0, "max":3000.0, "step":10.0}),
        }}
    def apply(self, audio, threshold, ratio, attack_ms, release_ms):
        return (_apply_board(audio, Pedalboard([Compressor(
            threshold_db=threshold, ratio=ratio,
            attack_ms=attack_ms, release_ms=release_ms)])),)


class ApplioGain:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":   ("AUDIO",),
            "gain_db": ("FLOAT", {"default": 0.0, "min":-60.0, "max":60.0, "step":1.0}),
        }}
    def apply(self, audio, gain_db):
        return (_apply_board(audio, Pedalboard([Gain(gain_db=gain_db)])),)


class ApplioLimiter:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":        ("AUDIO",),
            "threshold_db": ("FLOAT", {"default": -6.0, "min":-60.0, "max": 0.0,   "step": 1.0}),
            "release_ms":   ("FLOAT", {"default":100.0, "min":  1.0, "max":1000.0, "step":10.0}),
        }}
    def apply(self, audio, threshold_db, release_ms):
        return (_apply_board(audio, Pedalboard([Limiter(
            threshold_db=threshold_db, release_ms=release_ms)])),)


class ApplioPitchShiftFX:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":    ("AUDIO",),
            "semitones":("FLOAT", {"default": 0.0, "min":-24.0, "max":24.0, "step":0.5}),
        }}
    def apply(self, audio, semitones):
        return (_apply_board(audio, Pedalboard([PitchShift(semitones=semitones)])),)


class ApplioDistortion:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":    ("AUDIO",),
            "drive_db": ("FLOAT", {"default":25.0, "min":0.0, "max":100.0, "step":1.0}),
        }}
    def apply(self, audio, drive_db):
        return (_apply_board(audio, Pedalboard([Distortion(drive_db=drive_db)])),)


class ApplioBitcrush:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":     ("AUDIO",),
            "bit_depth": ("INT", {"default": 8, "min": 1, "max": 32}),
        }}
    def apply(self, audio, bit_depth):
        return (_apply_board(audio, Pedalboard([Bitcrush(bit_depth=bit_depth)])),)


class ApplioClipping:
    CATEGORY = "Audio/Applio/Effects"
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "apply"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":        ("AUDIO",),
            "threshold_db": ("FLOAT", {"default":-6.0, "min":-60.0, "max":0.0, "step":1.0}),
        }}
    def apply(self, audio, threshold_db):
        return (_apply_board(audio, Pedalboard([Clipping(threshold_db=threshold_db)])),)


# ─────────────────────────────────────────────────────────────
# 13. ApplioVoiceBlender
# ─────────────────────────────────────────────────────────────

class ApplioVoiceBlender:
    """두 .pth 모델을 블렌딩합니다. 출력 모델은 .index 없이 사용하세요."""

    CATEGORY = "Audio/Applio"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("message", "output_pth_path")
    FUNCTION      = "blend"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "output_name": ("STRING", {"default": "blended_model"}),
            "pth_path_a":  ("STRING", {"default": ""}),
            "pth_path_b":  ("STRING", {"default": ""}),
            "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "0=A만, 0.5=동일, 1=B만"}),
        }}

    def blend(self, output_name, pth_path_a, pth_path_b, ratio):
        try:
            message, out_path = model_blender(output_name, pth_path_a, pth_path_b, ratio)
            return (message, out_path or "")
        except Exception as e:
            msg = f"[Applio] Blend error: {e}"
            print(msg + "\n" + traceback.format_exc())
            return (msg, "")


# ─────────────────────────────────────────────────────────────
# 14. ApplioTTS
# ─────────────────────────────────────────────────────────────

class ApplioTTS:
    """
    EdgeTTS(Microsoft) 텍스트-음성 변환. 인터넷 연결 필요.

    voice 예시: en-US-AriaNeural, ja-JP-NanamiNeural,
                ko-KR-SunHiNeural, zh-CN-XiaoxiaoNeural
    """
    CATEGORY = "Audio/Applio"
    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "tts"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text":      ("STRING", {"default": "", "multiline": True}),
            "tts_voice": ("STRING", {"default": "en-US-AriaNeural"}),
            "speed":     ("INT",    {"default": 0, "min": -100, "max": 100,
                                     "tooltip": "속도 오프셋 (%)"}),
        }}

    def tts(self, text, tts_voice, speed):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = tmp.name; tmp.close()

        async def _run():
            rate = f"+{speed}%" if speed >= 0 else f"{speed}%"
            await edge_tts.Communicate(text, tts_voice, rate=rate).save(out_path)

        try:
            _run_async(_run())
            result = _from_wav(out_path)
        except Exception as e:
            print(f"[Applio] TTS error: {e}\n{traceback.format_exc()}")
            result = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 22050}
        finally:
            try: os.remove(out_path)
            except OSError: pass

        return (result,)


# ─────────────────────────────────────────────────────────────
# 15. ApplioAudioAnalyzer
# ─────────────────────────────────────────────────────────────

class ApplioAudioAnalyzer:
    """스펙트로그램 + 파형 + 스펙트럼 피처를 IMAGE로 반환합니다."""

    CATEGORY = "Audio/Applio"
    RETURN_TYPES  = ("STRING", "IMAGE")
    RETURN_NAMES  = ("info", "spectrogram")
    FUNCTION      = "analyze"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    def analyze(self, audio):
        in_path   = _to_wav(audio)
        plot_tmp  = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plot_path = plot_tmp.name; plot_tmp.close()

        try:
            info, _ = analyze_audio(in_path, save_plot_path=plot_path)
            from PIL import Image
            img = Image.open(plot_path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(arr).unsqueeze(0)
        except Exception as e:
            print(f"[Applio] Analyzer error: {e}\n{traceback.format_exc()}")
            info = f"error: {e}"
            image_tensor = torch.zeros(1, 64, 64, 3)
        finally:
            for p in (in_path, plot_path):
                try: os.remove(p)
                except OSError: pass

        return (info, image_tensor)


# ─────────────────────────────────────────────────────────────
# 등록
# ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ApplioModelLoader":   ApplioModelLoader,
    "ApplioInfer":         ApplioInfer,
    "ApplioReverb":        ApplioReverb,
    "ApplioChorus":        ApplioChorus,
    "ApplioDelay":         ApplioDelay,
    "ApplioCompressor":    ApplioCompressor,
    "ApplioGain":          ApplioGain,
    "ApplioLimiter":       ApplioLimiter,
    "ApplioPitchShiftFX":  ApplioPitchShiftFX,
    "ApplioDistortion":    ApplioDistortion,
    "ApplioBitcrush":      ApplioBitcrush,
    "ApplioClipping":      ApplioClipping,
    "ApplioVoiceBlender":  ApplioVoiceBlender,
    "ApplioTTS":           ApplioTTS,
    "ApplioAudioAnalyzer": ApplioAudioAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplioModelLoader":   "Applio Model Loader",
    "ApplioInfer":         "Applio Voice Convert",
    "ApplioReverb":        "Applio Reverb",
    "ApplioChorus":        "Applio Chorus",
    "ApplioDelay":         "Applio Delay",
    "ApplioCompressor":    "Applio Compressor",
    "ApplioGain":          "Applio Gain",
    "ApplioLimiter":       "Applio Limiter",
    "ApplioPitchShiftFX":  "Applio Pitch Shift FX",
    "ApplioDistortion":    "Applio Distortion",
    "ApplioBitcrush":      "Applio Bitcrush",
    "ApplioClipping":      "Applio Clipping",
    "ApplioVoiceBlender":  "Applio Voice Blender",
    "ApplioTTS":           "Applio TTS (EdgeTTS)",
    "ApplioAudioAnalyzer": "Applio Audio Analyzer",
}

"""
TTS model initialization and management
"""

import asyncio
import os
from enum import Enum
from typing import Any, Dict, List, Optional

import perth
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False

try:
    import safetensors.torch

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from app.config import Config, detect_device
from app.core.mtl import SUPPORTED_LANGUAGES

if not getattr(perth, "PerthImplicitWatermarker", None):
    print("⚠️  PerthImplicitWatermarker not found or broken. Applying mock to prevent crash.")

    class MockWatermarker:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, audio, *args, **kwargs):
            return audio

        def decode(self, audio, *args, **kwargs):
            return None

    perth.PerthImplicitWatermarker = MockWatermarker

_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_model_type = None
_is_multilingual = None
_supported_languages = {}

PARALINGUISTIC_TAGS = [
    {"tag": "[laugh]", "description": "Laughter"},
    {"tag": "[chuckle]", "description": "Light chuckle"},
    {"tag": "[cough]", "description": "Cough sound"},
    {"tag": "[sigh]", "description": "Sigh"},
    {"tag": "[gasp]", "description": "Gasp"},
    {"tag": "[clear throat]", "description": "Throat clearing"},
]


class ModelType(Enum):
    STANDARD = "standard"
    MULTILINGUAL = "multilingual"
    TURBO = "turbo"


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


async def initialize_model():
    global _model, _device, _initialization_state, _initialization_error
    global _initialization_progress, _model_type, _is_multilingual, _supported_languages

    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."

        Config.validate()
        _device = detect_device()

        model_type_str = Config.TTS_MODEL_TYPE
        print("Initializing Chatterbox TTS model...")
        print(f"Model type: {model_type_str}")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")

        _initialization_progress = "Creating model cache directory..."
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)

        _initialization_progress = "Checking voice sample..."
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(
                f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}"
            )

        _initialization_progress = "Configuring device compatibility..."
        if _device != "cuda":
            print(f"Applying torch.load patch for {_device} compatibility...")
            original_load = torch.load

            def force_cpu_torch_load(f, map_location=None, **kwargs):
                target_map = map_location if map_location is not None else "cpu"
                return original_load(f, map_location=target_map, **kwargs)

            torch.load = force_cpu_torch_load

            if HAS_SAFETENSORS:
                original_load_file = safetensors.torch.load_file

                def force_cpu_load_file(filename, device=None):
                    return original_load_file(filename, device="cpu")

                safetensors.torch.load_file = force_cpu_load_file

        _initialization_progress = "Loading TTS model (this may take a while)..."
        loop = asyncio.get_event_loop()

        def load_and_move_model(model_loader, target_device: str):
            model = model_loader()

            if target_device != "cpu":
                print(f"Moving model components to {target_device}...")
                for attr in ["t3", "s3gen", "ve"]:
                    if hasattr(model, attr):
                        component = getattr(model, attr)
                        if hasattr(component, "to"):
                            setattr(model, attr, component.to(target_device))

                if (
                    target_device == "mps"
                    and hasattr(torch, "mps")
                    and torch.mps.is_available()
                ):
                    torch.mps.empty_cache()

                model.device = target_device

            return model

        if model_type_str == "turbo":
            print("Loading Chatterbox Turbo TTS model...")
            _model = await loop.run_in_executor(
                None,
                lambda: load_and_move_model(
                    lambda: ChatterboxTurboTTS.from_pretrained(device="cpu"), _device
                ),
            )
            _model_type = ModelType.TURBO
            _is_multilingual = False
            _supported_languages = {"en": "English"}
            print(
                "Turbo model initialized (English only, paralinguistic tags supported)"
            )
        elif model_type_str == "multilingual":
            if not MULTILINGUAL_AVAILABLE:
                raise ImportError(
                    "Multilingual model not available. Install multilingual dependencies or use standard/turbo model."
                )
            print("Loading Chatterbox Multilingual TTS model...")
            _model = await loop.run_in_executor(
                None,
                lambda: load_and_move_model(
                    lambda: ChatterboxMultilingualTTS.from_pretrained(device="cpu"),
                    _device,
                ),
            )
            _model_type = ModelType.MULTILINGUAL
            _is_multilingual = True
            _supported_languages = SUPPORTED_LANGUAGES.copy()
            print(
                f"Multilingual model initialized with {len(_supported_languages)} languages"
            )
        else:
            print("Loading standard Chatterbox TTS model...")
            _model = await loop.run_in_executor(
                None,
                lambda: load_and_move_model(
                    lambda: ChatterboxTTS.from_pretrained(device="cpu"), _device
                ),
            )
            _model_type = ModelType.STANDARD
            _is_multilingual = False
            _supported_languages = {"en": "English"}
            print("Standard model initialized (English only)")

        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"Model initialized successfully on {_device}")
        return _model

    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"Failed to initialize model: {e}")
        raise e


def get_model():
    return _model


def get_device():
    return _device


def get_initialization_state():
    return _initialization_state


def get_initialization_progress():
    return _initialization_progress


def get_initialization_error():
    return _initialization_error


def is_ready():
    return (
        _initialization_state == InitializationState.READY.value and _model is not None
    )


def is_initializing():
    return _initialization_state == InitializationState.INITIALIZING.value


def is_multilingual():
    return _is_multilingual


def is_turbo():
    return _model_type == ModelType.TURBO


def get_model_type() -> Optional[str]:
    return _model_type.value if _model_type else None


def get_supported_languages():
    return _supported_languages.copy()


def supports_language(language_id: str):
    return language_id in _supported_languages


def get_paralinguistic_tags() -> List[Dict[str, str]]:
    if is_turbo():
        return PARALINGUISTIC_TAGS.copy()
    return []


def get_model_capabilities() -> Dict[str, Any]:
    return {
        "model_type": _model_type.value if _model_type else None,
        "supports_paralinguistic_tags": _model_type == ModelType.TURBO
        if _model_type
        else False,
        "supports_exaggeration": _model_type != ModelType.TURBO
        if _model_type
        else True,
        "supports_cfg_weight": _model_type != ModelType.TURBO if _model_type else True,
        "supports_temperature": _model_type != ModelType.TURBO if _model_type else True,
        "supported_languages": _supported_languages or {},
        "is_multilingual": _is_multilingual if _is_multilingual is not None else False,
        "paralinguistic_tags": PARALINGUISTIC_TAGS
        if _model_type == ModelType.TURBO
        else [],
    }


def get_model_info() -> Dict[str, Any]:
    return {
        "model_type": _model_type.value if _model_type else None,
        "is_multilingual": _is_multilingual if _is_multilingual is not None else False,
        "is_turbo": _model_type == ModelType.TURBO if _model_type else False,
        "supported_languages": _supported_languages or {},
        "language_count": len(_supported_languages) if _supported_languages else 0,
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state,
        "capabilities": get_model_capabilities(),
    }

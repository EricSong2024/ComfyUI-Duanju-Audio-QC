import json
import subprocess
import sys
import tempfile
from pathlib import Path

import folder_paths
import numpy as np
import soundfile as sf
import torch


class DuanjuCLAPBGMSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "positive_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "continuous instrumental suspense score; dark crime-thriller tension; restrained music under dialogue",
                    },
                ),
                "negative_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "vocals; singing; cheerful music; random impacts; thunder; sound-effects montage; harsh noise",
                    },
                ),
                "minimum_margin": (
                    "FLOAT",
                    {"default": 0.035, "min": -1.0, "max": 1.0, "step": 0.005},
                ),
            },
            "optional": {
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "BOOLEAN", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("selected_audio", "use_bgm", "selected_index", "score", "report_json")
    FUNCTION = "select"
    CATEGORY = "Duanju/Audio QC"

    def select(
        self,
        audio_1,
        positive_text,
        negative_text,
        minimum_margin,
        audio_2=None,
        audio_3=None,
        audio_4=None,
        audio_5=None,
        audio_6=None,
    ):
        inputs = [item for item in (audio_1, audio_2, audio_3, audio_4, audio_5, audio_6) if item is not None]
        sample_rate = int(audio_1["sample_rate"])
        candidates = []
        for item in inputs:
            waveform = item["waveform"].detach().cpu().float()
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim != 3:
                raise ValueError(
                    f"Expected AUDIO waveform [batch, channels, samples], got {tuple(waveform.shape)}"
                )
            if int(item["sample_rate"]) != sample_rate:
                raise ValueError("All candidate inputs must use the same sample rate")
            candidates.extend(waveform[index : index + 1] for index in range(waveform.shape[0]))

        model_dir = Path(folder_paths.models_dir) / "clap" / "clap-htsat-unfused"
        required = (
            "config.json",
            "merges.txt",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        )
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"CLAP model is incomplete at {model_dir}. Missing: {', '.join(missing)}"
            )

        scorer = Path(__file__).with_name("score_clap.py")
        with tempfile.TemporaryDirectory(prefix="duanju-clap-") as tmp:
            temp_dir = Path(tmp)
            audio_paths = []
            for index, candidate in enumerate(candidates):
                data = candidate.squeeze(0).numpy().T
                path = temp_dir / f"candidate_{index:03d}.wav"
                sf.write(path, data, sample_rate, subtype="FLOAT")
                audio_paths.append(str(path))

            request_path = temp_dir / "request.json"
            result_path = temp_dir / "result.json"
            request_path.write_text(
                json.dumps(
                    {
                        "model_dir": str(model_dir),
                        "audio_paths": audio_paths,
                        "positive_text": positive_text,
                        "negative_text": negative_text,
                        "minimum_margin": float(minimum_margin),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(scorer), str(request_path), str(result_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Duanju CLAP scorer failed.\n"
                    f"stdout:\n{completed.stdout[-4000:]}\n"
                    f"stderr:\n{completed.stderr[-4000:]}"
                )
            report = json.loads(result_path.read_text(encoding="utf-8"))

        selected_index = int(report["selected_index"])
        use_bgm = bool(report["use_bgm"])
        if use_bgm:
            selected = candidates[selected_index]
        else:
            selected = torch.zeros_like(candidates[0])

        return (
            {"waveform": selected, "sample_rate": sample_rate},
            use_bgm,
            selected_index,
            float(report["selected_score"]),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


NODE_CLASS_MAPPINGS = {"DuanjuCLAPBGMSelector": DuanjuCLAPBGMSelector}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DuanjuCLAPBGMSelector": "Duanju CLAP BGM Selector (CPU)"
}

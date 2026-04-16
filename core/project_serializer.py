import json
import os
from core.data_model import ProjectModel
from core.structures import Keyframe, EaseType, EffectType


class ProjectSerializer:
    @staticmethod
    def save_project(model: ProjectModel, filepath: str):
        data = {
            "version": "1.0",
            "video_path": model.video_path,
            "audio_path": model.audio_path,
            "audio_offset": model.audio_offset,
            "keyframes": []
        }

        # 序列化关键帧
        for kf in model.keyframes:
            kf_data = {
                "frame_idx": kf.frame_idx,
                "output_time": kf.output_time,
                "ease_type": kf.ease_type.name,  # Enum -> Str
                "effect_type": kf.effect_type.name,  # Enum -> Str
                "effect_params": kf.effect_params
            }
            data["keyframes"].append(kf_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_project(model: ProjectModel, filepath: str) -> dict:
        """
        加载工程数据到 model。
        注意：这只是加载数据，之后还需要触发 Video/Audio Loader 重新读取资源。
        返回一个 dict 包含需要重新加载的路径信息，供 Controller 使用。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Project file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 恢复基础属性
        model.audio_offset = data.get("audio_offset", 0.0)

        # 2. 恢复关键帧
        model.keyframes.clear()
        kf_list = data.get("keyframes", [])

        for kf_data in kf_list:
            kf = Keyframe(
                frame_idx=kf_data["frame_idx"],
                output_time=kf_data["output_time"]
            )
            # 恢复枚举
            if "ease_type" in kf_data:
                try:
                    kf.ease_type = EaseType[kf_data["ease_type"]]
                except:
                    pass

            if "effect_type" in kf_data:
                try:
                    kf.effect_type = EffectType[kf_data["effect_type"]]
                except:
                    pass

            kf.effect_params = kf_data.get("effect_params", {})
            model.keyframes.append(kf)

        # 排序以防万一
        model.keyframes.sort()
        model.keyframes_changed.emit()

        return {
            "video_path": data.get("video_path", ""),
            "audio_path": data.get("audio_path", "")
        }

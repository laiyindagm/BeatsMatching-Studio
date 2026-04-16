"""
engines/model_preloader.py
====================================
模型预加载器：在 GUI 启动时后台加载 ViTPose + HMR2 模型

设计目标：
  1. 软件打开即开始加载（不阻塞 UI）
  2. 加载完成后缓存模型实例
  3. 帧提取时复用缓存，跳过 5-10s 的加载时间
  4. 处理"正在加载时点击帧提取"的竞争条件

状态流转：
  idle → loading → ready (可复用)
                  → error   (下次重新加载)

线程安全说明：
  - 所有状态变量通过信号在主线程更新
  - 模型缓存使用 _model_cache 字典，仅在赋值后对外暴露
"""
from __future__ import annotations

import os
import sys
import time
import threading
from typing import Optional, Dict, Any, Callable

from PySide6.QtCore import QThread, Signal, QObject, QMutex, QWaitCondition, QMutexLocker

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class PreloadState:
    """预加载状态常量"""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelPreloaderWorker(QThread):
    """
    后台模型加载工作线程
    
    信号：
        model_loaded(str, float): 模型名称 + 加载耗时(ms)
        all_ready(): 所有模型加载完成
        error(str): 加载失败
    """
    
    model_loaded = Signal(str, float)  # model_name, load_time_s (秒)
    all_ready = Signal()
    error = Signal(str)
    
    def __init__(self, models_to_load: list[str] = None, parent=None):
        """
        Args:
            models_to_load: 要预加载的模型列表，如 ["vitpose", "hmr2"]。
                           None 或空列表表示全部预加载。
        """
        super().__init__(parent)
        self._models = models_to_load or ["vitpose", "hmr2"]
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            self.error.emit(f"预加载异常: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    
    def _run_impl(self):
        """执行模型预加载"""
        
        # 设置环境变量（必须在导入前）
        from pose_extraction.weights_config import (
            HF_HOME, VITPOSE_MODEL_ID, YOLO_WEIGHTS,
            HMR2_CHECKPOINT, SMPL_NEUTRAL,
        )
        os.environ["HF_HOME"] = HF_HOME
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        for model_name in self._models:
            if self._cancelled:
                return
            
            t0 = time.time()
            
            try:
                if model_name == "vitpose":
                    from pose_extraction.vitpose.detector import ViTPoseDetector
                    detector = ViTPoseDetector(
                        vitpose_model=VITPOSE_MODEL_ID,
                        yolo_model=YOLO_WEIGHTS,
                    )
                    # 线程安全写入全局缓存
                    with _cache_mutex:
                        _model_cache["vitpose_detector"] = detector
                    
                elif model_name == "hmr2":
                    from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
                    reconstructor = HMR2Reconstructor(
                        checkpoint_path=HMR2_CHECKPOINT,
                        yolo_model=YOLO_WEIGHTS,
                    )
                    with _cache_mutex:
                        _model_cache["hmr2_reconstructor"] = reconstructor
                
                elif model_name == "smpl":
                    from engines.smpl_renderer import SMPLRenderer
                    renderer = SMPLRenderer(SMPL_NEUTRAL)
                    with _cache_mutex:
                        _model_cache["smpl_renderer"] = renderer
                
                load_time_s = time.time() - t0
                self.model_loaded.emit(model_name, load_time_s)
                
            except Exception as e:
                import traceback
                self.error.emit(f"{model_name} 加载失败: {e}\n{traceback.format_exc()}")
                with _cache_mutex:
                    _model_cache[f"{model_name}_error"] = str(e)


# 全局模型缓存（线程安全通过主线程信号机制保证）
_model_cache: Dict[str, Any] = {}
_cache_mutex = threading.Lock()


class ModelPreloader(QObject):
    """
    模型预加载管理器（单例模式）
    
    使用方式：
        preloader = ModelPreloader.get_instance()
        preloader.start_preload(["vitpose", "hmr2"])
        preloader.model_ready.connect(on_ready)
        
        # 在需要时获取缓存的模型
        detector = preloader.get_model("vitpose_detector")
    """
    
    # 单例引用
    _instance: Optional['ModelPreloader'] = None
    
    # 信号
    state_changed = Signal(str, str)       # (model_name, new_state)
    model_loaded = Signal(str, float)     # (model_name, load_time_s) 秒
    all_models_ready = Signal()
    preload_error = Signal(str, str)      # (model_name, error_msg)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[ModelPreloaderWorker] = None
        self._states: Dict[str, str] = {}  # {model_name: state}
        self._load_times: Dict[str, float] = {}
    
    @classmethod
    def get_instance(cls) -> 'ModelPreloader':
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def start_preload(self, models: list[str] = None) -> None:
        """
        启动预加载（如果已在加载则不重复）
        
        Args:
            models: 要加载的模型列表，None=全部
        """
        if self._worker is not None and self._worker.isRunning():
            return  # 已在加载中
        
        self._worker = ModelPreloaderWorker(models)
        self._worker.model_loaded.connect(self._on_model_loaded)
        self._worker.all_ready.connect(self._on_all_ready)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        
        # 设置初始状态
        for m in (models or ["vitpose", "hmr2"]):
            self._set_state(m, PreloadState.LOADING)
    
    def _set_state(self, model_name: str, state: str) -> None:
        old_state = self._states.get(model_name)
        self._states[model_name] = state
        self.state_changed.emit(model_name, state)
    
    def _on_model_loaded(self, model_name: str, load_time_s: float) -> None:
        self._set_state(model_name, PreloadState.READY)
        self._load_times[model_name] = load_time_s
        self.model_loaded.emit(model_name, load_time_s)
    
    def _on_all_ready(self) -> None:
        self.all_models_ready.emit()
    
    def _on_error(self, msg: str) -> None:
        # 解析模型名
        parts = msg.split(None, 1)
        model_name = parts[0] if parts else "unknown"
        self._set_state(model_name, PreloadState.ERROR)
        self.preload_error.emit(model_name, msg)
    
    def is_ready(self, model_name: str) -> bool:
        """检查某个模型是否已就绪"""
        return self._states.get(model_name) == PreloadState.READY
    
    def is_loading(self, model_name: str) -> bool:
        return self._states.get(model_name) == PreloadState.LOADING
    
    def any_ready(self) -> bool:
        """是否有任何模型就绪"""
        return any(s == PreloadState.READY for s in self._states.values())
    
    def all_ready(self) -> bool:
        """是否所有请求的模型都就绪"""
        return len(self._states) > 0 and all(
            s == PreloadState.READY for s in self._states.values()
        )
    
    def get_state(self, model_name: str) -> str:
        return self._states.get(model_name, PreloadState.IDLE)
    
    def get_load_time(self, model_name: str) -> float:
        return self._load_times.get(model_name, 0)
    
    def get_model(self, key: str):
        """
        从缓存获取预加载的模型实例
        
        Args:
            key: 缓存键名，如 "vitpose_detector", "hmr2_reconstructor", "smpl_renderer"
        
        Returns:
            模型实例，或 None（如果未缓存）
        """
        with _cache_mutex:
            return _model_cache.get(key)
    
    def has_model(self, key: str) -> bool:
        with _cache_mutex:
            return key in _model_cache
    
    def clear_cache(self, key: str = None) -> None:
        """清除缓存（用于出错后重试）"""
        with _cache_mutex:
            if key:
                _model_cache.pop(key, None)
            else:
                _model_cache.clear()

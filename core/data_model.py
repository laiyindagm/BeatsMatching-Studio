# 存放 ProjectModel，管理所有数据状态
from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtGui import QUndoStack

from .commands import MoveKeyframeCommand, UpdatePropertyCommand
from .structures import Keyframe, KeyframeSource
from .structures import EaseType, EffectType


class ProjectModel(QObject):
    # 定义信号，当数据改变时通知 UI
    data_loaded = Signal()  # 视频/音频加载完成
    audio_data_loaded = Signal()
    keyframes_changed = Signal()  # 关键帧增删改或位置变化
    audio_offset_changed = Signal(float)  # 音频偏移量变化
    playhead_moved = Signal(float)  # 播放头位置改变
    pose_data_loaded = Signal()  # 姿态数据（ViTPose/HMR2）加载完成

    def __init__(self):
        super().__init__()
        # 基础数据
        self.video_path = ""
        self.fps = 30.0
        self.duration = 0.0
        self.frames = []  # 缓存的帧数据

        # 核心编辑数据
        self.keyframes: list[Keyframe] = []
        self.audio_path = ""
        self.audio_offset = 0.0  # 音频相对于视频开始的偏移(秒)
        self.audio_duration = 0.0
        self.beats = []  # 节拍时间点
        self.audio_waveform = []  # 用于绘制波形的数据

        # 姿态数据（由提取流程填充）
        self.pose_coords = None           # (T, 17, 2) ViTPose 平滑后关节坐标
        self.pose_rotations = None        # (T, 24, 3) HMR2 平滑后 SMPL 轴角
        self.pose_coords_raw = None       # (T, 17, 2) ViTPose 原始关节坐标
        self.pose_rotations_raw = None    # (T, 24, 3) HMR2 原始 SMPL 轴角
        self.pose_valid_mask_vit = None   # (T,) ViTPose 有效帧掩码
        self.pose_valid_mask_hmr = None   # (T,) HMR2 有效帧掩码

        # 状态数据
        self.current_time = 0.0
        self.undo_stack = QUndoStack(self)

        # 播放头信号节流 (16ms ≈ 60Hz 上限)
        self._playhead_throttle = QTimer(self)
        self._playhead_throttle.setSingleShot(True)
        self._playhead_throttle.setInterval(16)
        self._playhead_throttle.timeout.connect(self._emit_playhead)
        self._pending_time = None
        
        # 算法配置（由 GUI 算法配置面板设置）
        self.algo_config = None

    def set_pose_data(self, coords=None, rotations=None,
                      coords_raw=None, rotations_raw=None,
                      valid_vit=None, valid_hmr=None):
        """设置姿态数据并通知 UI"""
        self.pose_coords = coords
        self.pose_rotations = rotations
        self.pose_coords_raw = coords_raw
        self.pose_rotations_raw = rotations_raw
        self.pose_valid_mask_vit = valid_vit
        self.pose_valid_mask_hmr = valid_hmr
        self.pose_data_loaded.emit()

    def set_current_time(self, t: float):
        """统一设置当前时间，16ms 节流避免拖动时信号风暴"""
        if t < 0: t = 0.0
        if self.current_time != t:
            self.current_time = t
            self._pending_time = t
            if not self._playhead_throttle.isActive():
                # 立即发射第一次，后续 16ms 内的积累到下一次
                self.playhead_moved.emit(t)
                self._playhead_throttle.start()

    def set_current_time_immediate(self, t: float):
        """不节流版本，用于播放引擎 _on_tick 等高频路径"""
        if t < 0: t = 0.0
        if self.current_time != t:
            self.current_time = t
            self.playhead_moved.emit(t)

    def _emit_playhead(self):
        """节流定时器触发：发射积累的最后一次时间更新"""
        if self._pending_time is not None:
            self.playhead_moved.emit(self._pending_time)
            self._pending_time = None

    def set_audio_data(self, duration, waveform, beats):
        self.audio_duration = duration  # 注意区分 video duration
        self.audio_waveform = waveform  # np.array
        self.beats = beats  # list of float (seconds)
        self.audio_data_loaded.emit()  # 通知 UI 重绘

    def set_audio_offset(self, offset):
        self.audio_offset = offset
        self.audio_data_loaded.emit()

    def add_keyframe(self, frame_idx, time, source=None):
        # 逻辑：添加关键帧，排序，发送信号
        k = Keyframe(frame_idx=frame_idx, output_time=time)
        if source is not None:
            k.source = source
        self.keyframes.append(k)
        self.keyframes.sort()
        self.keyframes_changed.emit()

    def get_frame_at_time(self, t: float):
        # 将会调用 core/time_mapper.py 的算法
        pass

    def update_keyframe_time_preview(self, index: int, new_time: float):
        """只更新数据用于预览，不记录 Undo"""
        if 0 <= index < len(self.keyframes):
            self.keyframes[index].output_time = new_time
            self.keyframes_changed.emit()

    def update_keyframe_time(self, index: int, new_time: float, old_time: float):
        """
        更新指定索引关键帧的时间。
        注意：这里假设 index 不会因为排序而改变。
        实际上，如果你允许任意拖拽越过其他关键帧，需要先 pop 再 insert 并重新 sort。
        但我们在 UI 层限制了 min/max，所以顺序不会变，直接改值即可。
        """
        # 如果没变化，忽略
        if abs(new_time - old_time) < 1e-6: return

        # 创建命令并执行 (redo 会被自动调用)
        cmd = MoveKeyframeCommand(self, index, new_time, old_time)
        self.undo_stack.push(cmd)

    def select_keyframe(self, index: int, multi_select):
        """选中逻辑：单选"""
        if multi_select:
            self.keyframes[index].selected = not self.keyframes[index].selected
        else:
            for i, kf in enumerate(self.keyframes):
                kf.selected = (i == index)
        self.keyframes_changed.emit()

    def deselect_all_keyframes(self):
        for kf in self.keyframes:
            kf.selected = False
        self.keyframes_changed.emit()

    def batch_update_ease(self, new_ease_type):
        cmd = UpdatePropertyCommand(self, 'ease_type', new_ease_type, "Change Ease Type")
        self.undo_stack.push(cmd)

    def batch_update_vfx(self, new_effect_type):
        cmd = UpdatePropertyCommand(self, 'effect_type', new_effect_type, "Change VFX")
        self.undo_stack.push(cmd)

    def batch_update_effect_params(self, params: dict):
        cmd = UpdatePropertyCommand(self, 'effect_params', params, "Change VFX Params")
        self.undo_stack.push(cmd)

    def _batch_update(self, update_func):
        """内部辅助：遍历选中项并更新"""
        changed = False
        for kf in self.keyframes:
            if kf.selected:
                update_func(kf)
                changed = True

        if changed:
            self.keyframes_changed.emit()

    def delete_selected_keyframes(self):
        """删除选中关键帧（BOUNDARY 首尾帧不可删除，至少保留 1 个关键帧）"""
        original_len = len(self.keyframes)

        new_kfs = [kf for kf in self.keyframes
                   if not kf.selected or kf.source == KeyframeSource.BOUNDARY]

        # 至少保留 1 个关键帧
        if not new_kfs and self.keyframes:
            new_kfs = [self.keyframes[0]]

        if len(new_kfs) != original_len:
            self.keyframes = new_kfs
            self.keyframes_changed.emit()

    @staticmethod
    def get_ease_color(ease_type: EaseType):
        """变速连线颜色"""
        mapping = {
            EaseType.LINEAR: "#aaaaaa",  # 灰 (默认)
            EaseType.EASE_IN_QUAD: "#4caf50",  # 绿
            EaseType.EASE_OUT_QUAD: "#2196f3",  # 蓝
            EaseType.EASE_IN_OUT: "#9c27b0"  # 紫
        }
        return mapping.get(ease_type, "#aaaaaa")

    @staticmethod
    def get_effect_color(effect_type: EffectType):
        """动效关键帧颜色"""
        mapping = {
            EffectType.NONE: "#e0e0e0",  # 灰白 (无动效)
            EffectType.ZOOM_PUNCH: "#ff5722",  # 橙红
            EffectType.ROTATE_SHAKE: "#ffeb3b",  # 黄
            EffectType.FLASH: "#ffffff"
        }
        return mapping.get(effect_type, "#e0e0e0")




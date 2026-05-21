# gui/main_window.py
import os

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMainWindow, QSplitter, QFileDialog, QMessageBox, QWidget, QVBoxLayout, QApplication
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressBar, QLabel
from core.data_model import ProjectModel
from core.project_serializer import ProjectSerializer
from engines.audio_loader import AudioLoader
from engines.exporter import ExportWorker
from gui.timeline.timeline_container import TimelineContainer

from gui.player_widget import PlayerWidget
from engines.player_engine import PlayerEngine
from PySide6.QtWidgets import QToolBar
from gui.property_panel import PropertyPanel
from gui.pose_visualization_panel import PoseVisualizationPanel
from engines.algorithm_proxy import AlgorithmProxy
from gui.styles import (
    Icons, ModelStatusIndicator, DesignTokens,
    get_enhanced_stylesheet, apply_icon_to_action,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeatsMatching Studio")
        # 自适应屏幕大小
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.85), int(avail.height() * 0.85))
        else:
            self.resize(1200, 800)
        self.setMinimumSize(1024, 700)

        # 1. 核心模型
        self.model = ProjectModel()

        # 2. UI 组件
        self.timeline = TimelineContainer(self.model)

        # 初始化引擎
        self.engine = PlayerEngine(self.model)

        self.player_widget = PlayerWidget()

        self.engine.frame_ready.connect(self.player_widget.update_image)
        # 时间轴 -> 播放器跳转
        # 注意：TimeRuler 点击时调用的逻辑要改
        # 我们需要在 Model 里加个方法，或者直接在 View 层连接

        # 播放器 -> 时间轴游标同步
        self.engine.time_updated.connect(self.on_playback_time_updated)

        # 添加工具栏控制
        toolbar = QToolBar("Controls")
        self.addToolBar(toolbar)

        action_load = toolbar.addAction(Icons.LOAD_VIDEO + " 视频")
        action_load.triggered.connect(self.load_video_file)

        # 初始化 AudioLoader
        self.audio_loader = AudioLoader()

        # 连接 Loader -> Model
        self.audio_loader.audio_loaded.connect(self.on_audio_loaded)
        self.audio_loader.loading_progress.connect(self.on_audio_loading_started)

        toolbar = self.findChild(QToolBar)  # 或者是 self.addToolBar
        if not toolbar:
            toolbar = self.addToolBar("Main")

        act_load_audio = toolbar.addAction(Icons.LOAD_AUDIO + " 音频")
        act_load_audio.triggered.connect(self.load_audio_file)

        self.action_play = toolbar.addAction(Icons.PLAY + " 播放")
        self.action_play.triggered.connect(self.engine.toggle_play)
        self.action_play.setEnabled(False)  # 初始禁用，等视频加载后启用

        # 3. 布局
        # splitter = QSplitter(Qt.Vertical)
        # splitter.addWidget(self.player_widget)
        # splitter.addWidget(self.timeline)
        # splitter.setStretchFactor(0, 1)  # 播放器占大头
        # splitter.setStretchFactor(1, 0)  # 时间轴占小头
        #
        # self.setCentralWidget(splitter)
        # 1. 左侧区域 (预览 + 时间轴)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # 垂直 Splitter (预览 vs 时间轴)
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(self.player_widget)
        v_splitter.addWidget(self.timeline)
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        v_splitter.setSizes([600, 200])

        left_layout.addWidget(v_splitter)

        # 2. 右侧区域 (属性面板 + 位姿可视化面板)
        self.prop_panel = PropertyPanel(self.model)
        # 给面板设个最小/最大宽度，防止太扁或太宽
        self.prop_panel.setMinimumWidth(250)
        self.prop_panel.setMaximumWidth(400)
        self.prop_panel.setStyleSheet(f"""
            PropertyPanel {{
                background: {DesignTokens.BG_BASE};
                border-left: 1px solid {DesignTokens.BORDER_SUBTLE};
            }}
        """)

        # 位姿可视化面板（默认隐藏）
        self.pose_panel = PoseVisualizationPanel(self.model)
        self.pose_panel.setMinimumWidth(250)
        self.pose_panel.setMaximumWidth(400)
        self.pose_panel.setVisible(False)
        # 播放引擎直接驱动位姿面板（QueuedConnection 避免阻塞引擎定时器回调）
        self.engine.time_updated.connect(
            self.pose_panel.update_from_playback, Qt.ConnectionType.QueuedConnection)
        self.engine.playback_stopped.connect(self.pose_panel.notify_playback_stopped)

        # 右侧上下分割：属性面板 + 位姿面板
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.addWidget(self.prop_panel)
        self.right_splitter.addWidget(self.pose_panel)
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 1)
        self.right_splitter.setMinimumWidth(250)
        self.right_splitter.setMaximumWidth(400)

        # 3. 全局水平 Splitter
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.addWidget(left_widget)
        h_splitter.addWidget(self.right_splitter)
        h_splitter.setStretchFactor(0, 1)  # 左侧抢占空间
        h_splitter.setStretchFactor(1, 0)  # 右侧固定

        self.setCentralWidget(h_splitter)

        # 触发一次 range update，否则一开始滚动条没长度
        self.timeline.update_scroll_range()

        # 初始化状态栏
        self.status_bar = self.statusBar()

        # 创建进度条组件
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("extractionProgress")  # ID 选择器，确保 QSS 优先级
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)  # 默认隐藏

        self.status_label = QLabel("就绪")

        # 连接 VideoLoader 的信号
        # 注意：需要通过 PlayerEngine 访问 loader，或者把 loader 的信号透传出来
        self.engine.loader.loading_progress.connect(self.update_loading_progress)
        self.engine.loader.loading_finished.connect(self.on_loading_finished)
        self.engine.loader.meta_loaded.connect(self.on_meta_loaded)

        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.progress_bar)



        self.algo_proxy = AlgorithmProxy(self.model)

        # 连接算法代理进度/状态信号
        self.algo_proxy.extraction_progress.connect(self.update_loading_progress)
        self.algo_proxy.extraction_status.connect(self._on_extraction_status)

        # 工具栏添加算法按钮
        toolbar = self.findChild(QToolBar)
        toolbar.addSeparator()

        act_extract = toolbar.addAction(Icons.EXTRACT + " 提取")
        act_extract.triggered.connect(self._on_extract_keyframes)

        act_match = toolbar.addAction(Icons.AUTO_MATCH + " 匹配")
        act_match.triggered.connect(self._on_start_match)

        act_reset_speed = toolbar.addAction("↻ 重置变速")
        act_reset_speed.triggered.connect(self._on_reset_speed)

        # 导出 Worker
        self.export_worker = None

        # 添加菜单/按钮
        toolbar = self.findChild(QToolBar)
        toolbar.addSeparator()
        self.act_export = toolbar.addAction(Icons.EXPORT + " 导出")
        self.act_export.triggered.connect(self.start_export)

        # 创建 Undo/Redo Action
        # QUndoStack 提供了方便的 createUndoAction
        self.undo_action = self.model.undo_stack.createUndoAction(self, Icons.UNDO + " 撤销")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)  # Ctrl+Z

        self.redo_action = self.model.undo_stack.createRedoAction(self, Icons.REDO + " 重做")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)  # Ctrl+Shift+Z / Ctrl+Y

        # 添加到菜单栏
        toolbar.addSeparator()  # 添加分隔线
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)

        # 添加 Save/Load 按钮
        toolbar = self.findChild(QToolBar)
        toolbar.addSeparator()

        act_save = toolbar.addAction(Icons.SAVE + " 保存")
        act_save.triggered.connect(self.save_project)

        act_open = toolbar.addAction(Icons.OPEN + " 打开")
        act_open.triggered.connect(self.open_project)

        # 位姿可视化面板开关
        toolbar.addSeparator()
        self.act_pose_panel = toolbar.addAction("🦴 位姿")
        self.act_pose_panel.setCheckable(True)
        self.act_pose_panel.setChecked(False)
        self.act_pose_panel.triggered.connect(self._toggle_pose_panel)

        self.setStyleSheet(f"""
                    QSplitter::handle {{
                        background-color: {DesignTokens.BORDER_SUBTLE};
                    }}
                    QSplitter::handle:vertical {{
                        height: 3px;
                    }}
                    QSplitter::handle:horizontal {{
                        width: 4px;
                    }}
                    QSplitter::handle:hover {{
                        background-color: {DesignTokens.PRIMARY};
                    }}
                    QMainWindow {{
                        background-color: {DesignTokens.BG_DEEPEST};
                    }}
                """)

        # ── 模型状态指示器 ──
        self._model_status = ModelStatusIndicator()
        
        # 将状态指示器插入到状态栏最左侧
        self.status_bar.insertWidget(0, self._model_status)

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "project.json", "JSON Files (*.json)")
        if path:
            try:
                ProjectSerializer.save_project(self.model, path)
                self.status_bar.showMessage(f"Project saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Files (*.json)")
        if not path: return

        try:
            # 1. 加载数据
            paths = ProjectSerializer.load_project(self.model, path)
            video_path = paths["video_path"]
            audio_path = paths["audio_path"]

            # 2. 重新加载资源

            if video_path and os.path.exists(video_path):
                self.engine.load_video(video_path)
            else:
                QMessageBox.warning(self, "Warning", f"Video file not found:\n{video_path}")

            if audio_path and os.path.exists(audio_path):
                self.audio_loader.load_audio(audio_path)

            self.status_bar.showMessage(f"Project loaded from {path}", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{str(e)}")

    def start_export(self):
        if not self.model.video_path:
            QMessageBox.warning(self, "Warning", "No video loaded.")
            return

        # 选择保存路径
        path, _ = QFileDialog.getSaveFileName(self, "Export Video", "output.mp4", "MP4 Files (*.mp4)")
        if not path:
            return

        # 导出在 ExportWorker 后台线程中执行。不要禁用整个主窗口，否则
        # Qt 会把界面置灰，用户也无法继续查看时间轴或调整面板。
        self.act_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("导出中...")

        # 启动 Worker
        # 注意：需要传入 loader 以获取全帧缓存
        self.export_worker = ExportWorker(self.model, path, self.engine.loader)
        self.export_worker.progress.connect(self.update_export_progress)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.error.connect(self.on_export_error)
        self.export_worker.start()

    def update_export_progress(self, val):
        self.progress_bar.setValue(val)
        self.status_label.setText(f"导出中... {val}%")

    def on_export_finished(self, msg):
        self.act_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("就绪")
        QMessageBox.information(self, "成功", msg)
        self.export_worker = None

    def on_export_error(self, err):
        self.act_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("导出失败")
        QMessageBox.critical(self, "错误", f"导出失败:\n{err}")
        self.export_worker = None

    def load_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv)"
        )
        if path:
            self.engine.load_video(path)

    def on_playback_time_updated(self, t):
        # 更新时间轴的游标位置 (Scaler 不变，只是重绘)
        self.timeline.playhead.update()

        # 可选：自动跟随滚动 (如果游标跑出屏幕)
        # if t > self.timeline.scaler.get_visible_range()[1]: ...

    def on_meta_loaded(self, duration, w, h):
        self.status_label.setText("正在加载视频帧...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.action_play.setEnabled(False)
        # 此时可以禁用播放按钮

    def update_loading_progress(self, val):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(val)
        # 提取阶段只更新进度条，不覆盖 status_label（由 _on_extraction_status 管理）
        if not getattr(self, '_in_extraction', False):
            self.status_label.setText(f"加载中... {val}%")

    def on_loading_finished(self):
        self.progress_bar.setVisible(False)
        self.status_label.setText("视频加载完成")
        self.action_play.setEnabled(True)
        self.model.frames = self.engine.loader.frames_cache
        self.model.video_path = self.engine.loader.video_path
        # 加载完成后跳到第一帧
        if self.engine.loader.frames_cache:
            self.engine.seek(0)

    def load_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", "", "音频文件 (*.mp3 *.wav *.ogg *.flac)"
        )
        if path:
            # 启动波形分析
            self.audio_loader.load_audio(path)

    def on_audio_loading_started(self):
        self.status_bar.showMessage("Analyzing audio waveform...")

    def on_audio_loaded(self, duration, waveform, beats, full_audio):
        # 将分析结果存入 Model
        self.model.audio_path = self.audio_loader.path
        self.model.set_audio_data(duration, waveform, beats)
        self.engine.set_audio_data(full_audio)
        self.status_bar.showMessage("Audio Loaded", 3000)

    # ── 算法模式（从配置面板的 fusion_strategy 自动推导）─────────────────
    def _get_algo_mode(self) -> int:
        """
        从 model.algo_config.fusion_strategy 推导算法模式索引。
        0=Frame-Diff, 1=ViTPose, 2=HMR2, 3=Joint
        """
        strategy = getattr(getattr(self.model, 'algo_config', None),
                           'fusion_strategy', 'weighted_sum')
        strategy_map = {
            "vitpose_only": 1,
            "hmr2_only": 2,
            "weighted_sum": 3,
            "adaptive_alpha": 3,
            "pca_fusion": 3,
        }
        return strategy_map.get(strategy, 0)

    def _sync_algo_proxy_from_config(self):
        """根据当前算法配置同步 algo_proxy 的模式"""
        mode_idx = self._get_algo_mode()
        mode_map = {
            0: (False, "vitpose"),
            1: (True,  "vitpose"),
            2: (True,  "hmr2"),
            3: (True,  "joint"),
        }
        use_deep, mode = mode_map.get(mode_idx, (False, "vitpose"))
        config = getattr(self.model, 'algo_config', None)
        self.algo_proxy.set_use_deep_learning(use_deep, mode=mode, config=config)

    def _on_extract_keyframes(self):
        """启动关键帧提取（自动显示/隐藏进度条，含模型加载状态检查）"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 从配置推导模式并同步
        self._sync_algo_proxy_from_config()
        mode = self._get_algo_mode()
        
        # 深度学习模式：检查预加载状态
        if mode > 0:
            try:
                from engines.model_preloader import ModelPreloader, PreloadState
                preloader = ModelPreloader.get_instance()
                
                # 确定需要的模型
                needed_models = []
                if mode in (1, 3):  # ViTPose 或 Joint
                    needed_models.append("vitpose")
                if mode in (2, 3):  # HMR2 或 Joint
                    needed_models.append("hmr2")
                
                # 检查每个需要模型的状态
                for m in needed_models:
                    state = preloader.get_state(m)
                    if state == PreloadState.LOADING:
                        # 正在加载 → 显示提示，延迟启动或等待
                        self.status_label.setText(f"⏳ {m} 模型正在后台加载，请稍候...")
                        self.progress_bar.setRange(0, 0)  # 不定进度条
                        
                        # 使用 QTimer 延迟重试（最多等 60 秒）
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(2000, lambda: self._retry_extraction_if_ready())
                        return
                    
                    elif state == PreloadState.ERROR:
                        # 加载失败 → 清除缓存以便重新加载
                        preloader.clear_cache(m)
                        self.status_label.setText(f"⚠️ {m} 模型加载失败，正在重新加载...")
                        preloader.start_preload([m])
                        QTimer.singleShot(3000, lambda: self._retry_extraction_if_ready())
                        return
                
                    elif state == PreloadState.IDLE:
                        # 未开始预加载 → 立即开始
                        preloader.start_preload(needed_models)
                        self.status_label.setText(f"⏳ 首次加载 {needed_models} 模型...")
                        QTimer.singleShot(1000, lambda: self._on_extract_keyframes())
                        return
                        
                    # PreloadState.READY → 继续执行正常流程
                    
            except Exception:
                pass  # preloader 不可用时直接执行
        
        # 正常启动提取
        self._in_extraction = True
        self.algo_proxy.start_keyframe_extraction()
    
    def _retry_extraction_if_ready(self):
        """如果模型就绪则重试提取"""
        try:
            from engines.model_preloader import ModelPreloader, PreloadState
            preloader = ModelPreloader.get_instance()
            
            mode = self._get_algo_mode()
            needed = []
            if mode in (1, 3): needed.append("vitpose")
            if mode in (2, 3): needed.append("hmr2")
            
            all_ok = all(preloader.get_state(m) == PreloadState.READY for m in needed)
            
            if all_ok:
                self._on_extract_keyframes()  # 递归调用，这次会走正常路径
            else:
                # 还有未就绪的，继续等
                not_ready = [m for m in needed if preloader.get_state(m) != PreloadState.READY]
                self.status_label.setText(f"⏳ 等待 {', '.join(not_ready)} 加载完成...")
                from PySide6.QtCore import QTimer
                QTimer.singleShot(2000, self._retry_extraction_if_ready)
                
        except Exception as e:
            self.status_label.setText(f"状态检查异常，尝试直接启动...")
            self.algo_proxy.start_keyframe_extraction()
    
    def _on_start_match(self):
        """匹配按钮包装：显示进度条 + 启动匹配"""
        self.progress_bar.setRange(0, 0)  # 不定进度
        self.progress_bar.setVisible(True)
        self.algo_proxy.start_auto_match()

    def _on_reset_speed(self):
        """重置变速：将所有关键帧恢复到原始时间"""
        if not self.model.keyframes:
            self.status_label.setText("⚠️ 无关键帧可重置")
            return
        self.algo_proxy.reset_speed()

    def _on_extraction_status(self, message: str):
        """处理提取状态更新，同时更新模型指示器"""
        self.status_label.setText(message)
        
        # 提取/匹配完成或缺失数据 → 恢复进度条，清除提取标记
        if "提取完成" in message or "匹配完成" in message or "无法匹配" in message:
            self._in_extraction = False
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
        elif "错误" in message or "error" in message.lower():
            self._in_extraction = False
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setVisible(False)
        
        # 根据状态消息自动更新模型状态
        msg_lower = message.lower()
        if "加载" in msg_lower and ("vitpose" in msg_lower or "模型" in msg_lower):
            pass  # 保持 loading 状态
        elif "推理" in msg_lower:
            if "vitpose" in msg_lower or "hmr2" not in msg_lower:
                self._model_status.set_model_ready("ViTPose")
            if "hmr2" in msg_lower:
                self._model_status.set_model_ready("HMR2")
        elif "完成" in msg_lower or "error" in msg_lower or "错误" in msg_lower:
            if "error" in msg_lower or "错误" in msg_lower:
                self._model_status.set_model_error("Model")
            else:
                # 提取完成，标记所有模型就绪
                for m in ["ViTPose", "HMR2"]:
                    self._model_status.set_model_ready(m)

    def _toggle_pose_panel(self, checked: bool):
        """切换位姿可视化面板显示/隐藏"""
        self.pose_panel.setVisible(checked)
        if checked:
            # 确保 pose_panel 在 splitter 中获得合理高度
            total_h = self.right_splitter.height()
            if total_h > 0:
                self.right_splitter.setSizes([total_h // 2, total_h // 2])
            # 若已有 pose 数据，延迟一帧后渲染（等待布局完成）
            if self.model.pose_coords is not None or self.model.pose_rotations is not None:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(50, lambda: self.pose_panel._update_frame(self.model.current_time))

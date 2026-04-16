"""
gui/styles.py
====================================
BeatsMatching GUI 增强样式系统
在 qt_material 主题基础上叠加精细化样式，提升质感

包含：
  - 设计令牌系统 (DesignTokens)
  - 工具栏图标化（Unicode 符号作为轻量级图标）
  - 模型状态指示器
  - 状态栏美化
  - 控件微调
"""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSizePolicy, QFrame
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette


# ══════════════════════════════════════════════════════════════════════════════
#  设计令牌 (Design Tokens) — 统一的视觉规范
# ══════════════════════════════════════════════════════════════════════════════
class DesignTokens:
    """
    集中定义所有设计常量，蓝灰色调暗黑专业剪辑风。
    所有组件应引用此类而非硬编码颜色值。
    """
    # ── 背景层级 (从深到浅) ──
    BG_DEEPEST    = "#0D1017"   # 最深层 (app 背景)
    BG_BASE       = "#111620"   # 基础面板背景
    BG_SURFACE    = "#161B26"   # 主要面板 / 轨道背景
    BG_ELEVATED   = "#1C2230"   # 悬浮面板 / 卡片
    BG_OVERLAY    = "#232A3A"   # 弹出层 / tooltip

    # ── 品牌色 (Teal 系) ──
    PRIMARY       = "#26a69a"   # 常态品牌色
    PRIMARY_LIGHT = "#33CCBB"   # 亮态 / hover
    PRIMARY_DIM   = "#1B7A70"   # 暗态 / pressed
    PRIMARY_GHOST = "rgba(38,166,154,0.12)"  # 幽灵态背景

    # ── 强调色 ──
    ACCENT_BLUE   = "#5B8CFF"   # 播放头、选中高亮
    ACCENT_BLUE_DIM = "rgba(91,140,255,0.25)"  # 蓝色 glow
    ACCENT_RED    = "#FF6B6B"   # 节拍线
    ACCENT_RED_DIM = "rgba(255,107,107,0.30)"  # 红色 glow
    ACCENT_ORANGE = "#FFB84D"   # 选中关键帧
    ACCENT_GREEN  = "#4CAF88"   # 成功/就绪
    ACCENT_YELLOW = "#FFAA00"   # 加载中/警告

    # ── 文字层级 ──
    TEXT_PRIMARY   = "#E5E7EB"  # 主文字
    TEXT_SECONDARY = "#9CA3AF"  # 次级文字
    TEXT_DISABLED  = "#6B7280"  # 禁用态文字
    TEXT_INVERSE   = "#FFFFFF"  # 反色（品牌色背景上）

    # ── 边框与分割 ──
    BORDER_SUBTLE  = "rgba(255,255,255,0.06)"  # 极淡分割线
    BORDER_DEFAULT = "rgba(255,255,255,0.10)"  # 默认边框
    BORDER_STRONG  = "rgba(255,255,255,0.16)"  # 明显边框 / hover

    # ── 间距系统 ──
    SPACE_XS = 4
    SPACE_SM = 8
    SPACE_MD = 12
    SPACE_LG = 16
    SPACE_XL = 24

    # ── 圆角系统 ──
    RADIUS_SM = 4
    RADIUS_MD = 6
    RADIUS_LG = 8
    RADIUS_XL = 12

    # ── 时间轴专用 ──
    TL_WAVEFORM_TOP    = "#22c55e"   # 波形渐变顶部
    TL_WAVEFORM_BOTTOM = "#0d7a3e"   # 波形渐变底部
    TL_CLIP_COLOR      = "#2A6EBB"   # 视频条颜色
    TL_CLIP_HIGHLIGHT  = "rgba(255,255,255,0.08)"  # 视频条顶部高光
    TL_PLAYHEAD        = "#5B8CFF"   # 播放头 (蓝色，与节拍红线区分)
    TL_BEAT            = "#FF6B6B"   # 节拍红线
    TL_KF_DEFAULT      = "#D4D4D8"   # 关键帧默认
    TL_KF_SELECTED     = "#FFB84D"   # 关键帧选中
    TL_KF_HOVER        = "#FFFFFF"   # 关键帧hover


# ══════════════════════════════════════════════════════════════════════════════
#  图标常量（使用 Unicode 符号，无需外部资源文件）
# ══════════════════════════════════════════════════════════════════════════════
class Icons:
    """工具栏 Unicode 图标"""
    LOAD_VIDEO = "🎬"
    LOAD_AUDIO = "🎵"
    PLAY = "▶"
    PAUSE = "⏸"
    EXTRACT = "🔍"       # 关键帧提取
    AUTO_MATCH = "🔗"     # 自动匹配
    EXPORT = "💾"         # 导出视频
    UNDO = "↩"
    REDO = "↪"
    SAVE = "💾"
    OPEN = "📂"

    # 状态指示
    READY = "✅"
    LOADING = "⏳"
    ERROR = "❌"
    WARNING = "⚠️"
    GPU_ACTIVE = "🟢"
    GPU_INACTIVE = "⚫"


# ══════════════════════════════════════════════════════════════════════════════
#  模型状态指示器 Widget
# ══════════════════════════════════════════════════════════════════════════════
class ModelStatusIndicator(QWidget):
    """
    显示模型加载状态的精致状态条

    特性：
      - 圆角胶囊状背景
      - 动态颜色（加载中=橙色, 就绪=绿色, 未加载=灰色）
      - 脉冲动画效果（加载中时）
      - 显示模型名称和耗时
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        # 状态图标
        self._icon_label = QLabel(Icons.LOADING)
        self._icon_label.setFixedSize(16, 16)
        self._icon_label.setAlignment(Qt.AlignCenter)
        self._icon_label.setStyleSheet("font-size: 12px; background: transparent;")

        # 状态文字
        self._status_label = QLabel("模型未加载")
        self._status_label.setStyleSheet(f"""
            font-size: 11px;
            color: {DesignTokens.TEXT_DISABLED};
            background: transparent;
            font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
        """)

        # 详情标签（显示耗时）
        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet(f"""
            font-size: 10px;
            color: {DesignTokens.TEXT_DISABLED};
            background: transparent;
        """)

        layout.addWidget(self._icon_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._detail_label)
        layout.addStretch()

        # 脉冲动画定时器（必须在 _set_state 之前初始化）
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_animation)
        self._pulse_phase = 0

        # 状态追踪
        self._models = {}   # {"vitpose": (state, time_ms), "hmr2": ...}

        # 初始状态：未加载
        self._set_state("unloaded", "点击「帧提取」自动加载模型")

    def _set_state(self, state: str, message: str, detail: str = ""):
        """设置视觉状态"""
        DT = DesignTokens
        styles = {
            "unloaded": f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {DT.BG_ELEVATED}, stop:1 {DT.BG_SURFACE});
                border: 1px solid {DT.BORDER_DEFAULT};
                border-radius: 14px;
            """,
            "loading": f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3A2E18, stop:1 #2A2010);
                border: 1px solid #7A6330;
                border-radius: 14px;
            """,
            "ready": f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #142E22, stop:0.5 #10281C, stop:1 #0C2016);
                border: 1px solid #1E5A3A;
                border-radius: 14px;
            """,
            "error": f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2E1418, stop:1 #241014);
                border: 1px solid #6A2A2A;
                border-radius: 14px;
            """,
        }

        icon_map = {
            "unloaded": Icons.GPU_INACTIVE,
            "loading": Icons.LOADING,
            "ready": Icons.READY,
            "error": Icons.ERROR,
        }

        color_map = {
            "unloaded": DT.TEXT_DISABLED,
            "loading": DT.ACCENT_YELLOW,
            "ready": DT.ACCENT_GREEN,
            "error": DT.ACCENT_RED,
        }

        self.setStyleSheet(styles.get(state, styles["unloaded"]))
        self._icon_label.setText(icon_map.get(state, Icons.WARNING))
        self._status_label.setText(message)
        self._status_label.setStyleSheet(f"""
            font-size: 11px; font-weight: bold;
            color: {color_map.get(state, DT.TEXT_DISABLED)};
            background: transparent;
            font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
        """)

        if detail:
            self._detail_label.setText(detail)
            self._detail_label.show()
        else:
            self._detail_label.hide()

        # 加载中启动脉冲
        if state == "loading":
            if not self._pulse_timer.isActive():
                self._pulse_timer.start(500)
        else:
            self._pulse_timer.stop()
            self._pulse_phase = 0

    def _pulse_animation(self):
        """脉冲效果：图标透明度渐变"""
        self._pulse_phase = (self._pulse_phase + 1) % 6
        alpha = 0.3 + 0.7 * abs(3 - self._pulse_phase) / 3
        r = int(255 * alpha)
        g = int(170 * alpha)
        b = int(0 * alpha)
        self._icon_label.setStyleSheet(f"""
            font-size: 12px; color: rgb({r},{g},{b});
            background: transparent;
        """)
    
    def set_model_loading(self, model_name: str):
        """设置某个模型正在加载"""
        self._models[model_name] = ("loading", 0)
        self._update_combined_state()
    
    def set_model_ready(self, model_name: str, load_time_ms: float = 0):
        """设置某个模型已就绪"""
        self._models[model_name] = ("ready", load_time_ms)
        self._update_combined_state()
    
    def set_model_error(self, model_name: str, error: str = ""):
        """设置模型加载失败"""
        self._models[model_name] = ("error", 0)
        self._update_combined_state()
    
    def _update_combined_state(self):
        """根据所有模型的状态更新整体显示"""
        if not self._models:
            self._set_state("unloaded", "模型未加载")
            return
        
        names = list(self._models.keys())
        loading = [n for n in names if self._models[n][0] == "loading"]
        ready = [n for n in names if self._models[n][0] == "ready"]
        errors = [n for n in names if self._models[n][0] == "error"]
        
        if errors:
            self._set_state("error", f"{', '.join(errors)} 加载失败")
        elif loading:
            self._set_state("loading", f"正在加载 {', '.join(loading)}...")
        elif ready:
            total_time = sum(self._models[n][1] for n in ready)
            detail = f"({total_time:.1f}s)" if total_time > 0 else ""
            self._set_state("ready", f"{Icons.GPU_ACTIVE} {', '.join(ready)} 已就绪", detail)
        else:
            self._set_state("unloaded", "模型未加载")


# ══════════════════════════════════════════════════════════════════════════════
#  全局增强样式表
# ══════════════════════════════════════════════════════════════════════════════
def get_enhanced_stylesheet() -> str:
    """
    返回增强版样式表，覆盖 qt_material 的默认值以获得更精致的质感。
    基于 DesignTokens 蓝灰色调暗黑专业剪辑风。

    使用方式：
        app.setStyleSheet(app.styleSheet() + get_enhanced_stylesheet())
    注意要在 apply_stylesheet() 之后调用此函数的结果做追加。
    """
    DT = DesignTokens
    return f"""
    /* ── 全局字体渲染优化 ── */
    QWidget {{
        font-family: 'Microsoft YaHei UI', 'Segoe UI', 'PingFang SC', sans-serif;
        letter-spacing: 0.2px;
    }}

    /* ── 主窗口背景 ── */
    QMainWindow {{
        background-color: {DT.BG_DEEPEST};
    }}

    /* ── 工具栏增强 ── */
    QToolBar {{
        spacing: 4px;
        padding: 4px 8px;
        border: none;
        border-bottom: 1px solid {DT.BORDER_SUBTLE};
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {DT.BG_ELEVATED}, stop:1 {DT.BG_SURFACE});
    }}
    QToolBar QToolButton {{
        padding: 6px 12px;
        border-radius: {DT.RADIUS_MD}px;
        border: 1px solid transparent;
        font-size: 12px;
        font-weight: 600;
        min-width: 60px;
        background: transparent;
        color: {DT.TEXT_PRIMARY};
    }}
    QToolBar QToolButton:hover {{
        background: rgba(255,255,255,0.06);
        border: 1px solid {DT.BORDER_DEFAULT};
        color: {DT.TEXT_INVERSE};
    }}
    QToolBar QToolButton:pressed {{
        background: {DT.PRIMARY_GHOST};
        border: 1px solid {DT.PRIMARY};
        color: {DT.PRIMARY_LIGHT};
    }}
    QToolBar QToolButton:disabled {{
        color: {DT.TEXT_DISABLED};
        background: transparent;
    }}
    QToolBar QToolButton:checked {{
        background: {DT.PRIMARY_GHOST};
        border: 1px solid {DT.PRIMARY};
        color: {DT.PRIMARY_LIGHT};
    }}
    QToolBar QToolButton:checked:hover {{
        background: rgba(38, 166, 154, 0.2);
        border: 1px solid {DT.PRIMARY_LIGHT};
    }}

    /* ── 主操作按钮 (Export 等) ── */
    QToolBar QToolButton#PrimaryAction {{
        background: {DT.PRIMARY};
        color: {DT.TEXT_INVERSE};
        border: 1px solid {DT.PRIMARY_LIGHT};
        font-weight: bold;
    }}
    QToolBar QToolButton#PrimaryAction:hover {{
        background: {DT.PRIMARY_LIGHT};
    }}
    QToolBar QToolButton#PrimaryAction:pressed {{
        background: {DT.PRIMARY_DIM};
    }}

    /* ── 分隔线增强 ── */
    QToolBar::separator {{
        width: 1px;
        margin: 4px 8px;
        background: {DT.BORDER_SUBTLE};
    }}

    /* ── 下拉框增强 ── */
    QComboBox {{
        padding: 4px 12px 4px 8px;
        border-radius: {DT.RADIUS_MD}px;
        border: 1px solid {DT.BORDER_DEFAULT};
        background: {DT.BG_SURFACE};
        color: {DT.TEXT_PRIMARY};
        min-width: 120px;
    }}
    QComboBox:hover {{
        border: 1px solid {DT.BORDER_STRONG};
    }}
    QComboBox:focus {{
        border: 1px solid {DT.PRIMARY};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
        subcontrol-origin: padding;
        subcontrol-position: right center;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid {DT.TEXT_SECONDARY};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {DT.BG_ELEVATED};
        border: 1px solid {DT.BORDER_DEFAULT};
        selection-background-color: {DT.PRIMARY};
        color: {DT.TEXT_PRIMARY};
        outline: none;
        padding: 4px;
    }}

    /* ── 状态栏增强 ──*/
    QStatusBar {{
        background: {DT.BG_BASE};
        border-top: 1px solid {DT.BORDER_SUBTLE};
        font-size: 11px;
        color: {DT.TEXT_SECONDARY};
        padding: 2px 8px;
    }}
    QStatusBar QLabel {{
        color: {DT.TEXT_SECONDARY};
        background: transparent;
    }}
    QStatusBar::item {{
        border: none;
    }}

    /* ── 进度条增强（ID 选择器，确保覆盖 qt_material）── */
    QProgressBar#extractionProgress {{
        border: 1px solid {DT.BORDER_STRONG};
        border-radius: {DT.RADIUS_MD}px;
        background: {DT.BG_BASE};
        text-align: center;
        color: {DT.TEXT_PRIMARY};
        font-size: 11px;
        font-weight: bold;
        min-height: 20px;
        padding: 1px;
    }}
    QProgressBar#extractionProgress::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {DT.PRIMARY_LIGHT}, stop:0.5 {DT.PRIMARY}, stop:1 {DT.PRIMARY_DIM});
        border-radius: {DT.RADIUS_SM}px;
        min-width: 8px;
    }}

    /* ── Splitter 增强 ──*/
    QSplitter::handle {{
        background: {DT.BORDER_SUBTLE};
    }}
    QSplitter::handle:horizontal {{
        width: 4px;
    }}
    QSplitter::handle:vertical {{
        height: 3px;
    }}
    QSplitter::handle:hover {{
        background: {DT.PRIMARY};
    }}

    /* ── GroupBox 增强 ──*/
    QGroupBox {{
        font-weight: bold;
        font-size: 11px;
        border: 1px solid {DT.BORDER_DEFAULT};
        border-radius: {DT.RADIUS_LG}px;
        margin-top: 8px;
        padding-top: 20px;
        background: {DT.BG_SURFACE};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {DT.PRIMARY_LIGHT};
    }}

    /* ── TabWidget 增强 ──*/
    QTabWidget::pane {{
        border: 1px solid {DT.BORDER_SUBTLE};
        border-radius: {DT.RADIUS_LG}px;
        background: {DT.BG_BASE};
        top: -1px;
    }}
    QTabBar::tab {{
        padding: 8px 16px;
        border-radius: {DT.RADIUS_SM}px {DT.RADIUS_SM}px 0 0;
        margin-right: 2px;
        font-size: 11px;
        color: {DT.TEXT_SECONDARY};
        background: transparent;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:selected {{
        background: {DT.BG_SURFACE};
        color: {DT.PRIMARY_LIGHT};
        border-bottom: 2px solid {DT.PRIMARY};
    }}
    QTabBar::tab:hover:!selected {{
        background: rgba(255,255,255,0.04);
        color: {DT.TEXT_PRIMARY};
    }}

    /* ── SpinBox 增强 ──*/
    QSpinBox, QDoubleSpinBox {{
        padding: 4px 8px;
        border: 1px solid {DT.BORDER_DEFAULT};
        border-radius: {DT.RADIUS_SM}px;
        background: {DT.BG_SURFACE};
        color: {DT.TEXT_PRIMARY};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {DT.PRIMARY};
    }}

    /* ── CheckBox 增强 ──*/
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid {DT.BORDER_STRONG};
        background: {DT.BG_SURFACE};
    }}
    QCheckBox::indicator:checked {{
        background: {DT.PRIMARY};
        border: 1px solid {DT.PRIMARY};
    }}

    /* ── Button 增强（通用） ──*/
    QPushButton {{
        padding: 6px 18px;
        border-radius: {DT.RADIUS_MD}px;
        border: 1px solid {DT.BORDER_DEFAULT};
        font-weight: 500;
        font-size: 11px;
        background: {DT.BG_ELEVATED};
        color: {DT.TEXT_PRIMARY};
    }}
    QPushButton:hover {{
        border: 1px solid {DT.BORDER_STRONG};
        background: rgba(255,255,255,0.06);
    }}
    QPushButton:pressed {{
        background: {DT.PRIMARY_GHOST};
        border: 1px solid {DT.PRIMARY};
    }}

    /* ── ScrollBar 增强（更窄更精致） ──*/
    QScrollBar:vertical {{
        width: 6px;
        background: {DT.BG_BASE};
        border-radius: 3px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: rgba(255,255,255,0.12);
        border-radius: 3px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: rgba(255,255,255,0.20);
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: {DT.BG_BASE};
    }}

    QScrollBar:horizontal {{
        height: 6px;
        background: {DT.BG_BASE};
        border-radius: 3px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: rgba(255,255,255,0.12);
        border-radius: 3px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: rgba(255,255,255,0.20);
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ── Label 标题样式 ──*/
    QLabel#SectionTitle {{
        font-weight: 600;
        font-size: 13px;
        color: {DT.TEXT_PRIMARY};
        padding: 4px 0;
    }}
"""


def apply_icon_to_action(action, icon_text: str) -> None:
    """给 QAction 设置带图标的文字"""
    action.setText(f"{icon_text}  {action.text().strip()}")

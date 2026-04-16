# main.py

import sys
import os
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet
from PySide6.QtGui import QFont, QIcon
import ctypes
# 引入主窗口
from gui.main_window import MainWindow

# 解决高DPI缩放问题 (Windows)
os.environ["QT_FONT_DPI"] = "96"


def set_windows_dark_titlebar(window):
    """
    [Windows Only] 强制将窗口标题栏设置为深色模式
    """
    try:
        # 1. 检查是否是 Windows 系统
        if sys.platform != "win32":
            return

        # 2. 获取窗口句柄 (HWND)
        # PySide6 中获取 HWND 的方式:
        hwnd = window.winId()

        # 3. 调用 DWM API
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 11 build 22000+)
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 19 (Windows 10, older builds)

        dwm = ctypes.windll.dwmapi

        # 尝试设置属性 20 (适用于 Win11 和较新 Win10)
        value = ctypes.c_int(1)
        result = dwm.DwmSetWindowAttribute(int(hwnd), 20, ctypes.byref(value), ctypes.sizeof(value))

        # 如果失败，尝试属性 19 (适用于旧版 Win10)
        if result != 0:
            dwm.DwmSetWindowAttribute(int(hwnd), 19, ctypes.byref(value), ctypes.sizeof(value))

    except Exception as e:
        print(f"Failed to set dark title bar: {e}")


def set_app_icon(app, icon_path):
    """
    设置任务栏图标和窗口图标
    """
    if os.path.exists(icon_path):
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)

        # [Windows Only] 解决任务栏显示 Python 默认图标的问题
        # Windows 会把所有 python 脚本归为同一个组，需要设置独立的 AppUserModelID
        if sys.platform == 'win32':
            myappid = 'mycompany.beatsmatching.studio.1.0'  # 任意唯一的字符串
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(__file__), 'resources', 'icon.jpg')
    set_app_icon(app, icon_path)

    # 1. 设置全局字体（Microsoft YaHei UI 中英文渲染均优）
    font = QFont("Microsoft YaHei UI", 9)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    # 2. 应用 Material 主题
    extra = {
        'density_scale': '-1',
    }
    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)
    
    # 2.5 在 app 级别追加自定义样式（确保优先级高于 qt_material 默认值）
    # 注意：必须在 apply_stylesheet 之后追加，否则会被覆盖
    from gui.styles import get_enhanced_stylesheet
    app.setStyleSheet(app.styleSheet() + get_enhanced_stylesheet())

    # 3. 启动主窗口
    window = MainWindow()
    set_windows_dark_titlebar(window)
    window.show()

    # 4. 后台预加载深度学习模型（不阻塞 UI）
    try:
        from engines.model_preloader import ModelPreloader
        preloader = ModelPreloader.get_instance()
        preloader.model_loaded.connect(lambda name, t_s: print(
            f"[Preload] {name} 模型就绪 ({t_s:.1f}s)"))
        preloader.preload_error.connect(lambda name, err: print(
            f"[Preload] {err}"))
        preloader.all_models_ready.connect(
            lambda: print("[Preload] ✅ 所有模型已预加载完成！"))
        preloader.start_preload(["vitpose", "hmr2"])
        
        # 连接预加载状态到主窗口的模型状态指示器
        _name_map = {"vitpose": "ViTPose", "hmr2": "HMR2"}
        def _on_preload_loaded(name, t_s):
            display_name = _name_map.get(name, name.capitalize())
            window._model_status.set_model_ready(display_name, load_time_ms=t_s * 1000)
        preloader.model_loaded.connect(_on_preload_loaded)
        
        def _on_preload_error(name, err):
            display_name = _name_map.get(name, name.capitalize())
            window._model_status.set_model_error(display_name)
        preloader.preload_error.connect(_on_preload_error)
        
    except Exception as e:
        print(f"[Preload] 预加载未启用: {e}")

    sys.exit(app.exec())

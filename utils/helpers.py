def format_time(seconds: float) -> str:
    """
    将秒数转换为 MM:SS 或 MM:SS.ms 格式
    例如:
    65.5 -> "01:05.5"
    120 -> "02:00"
    """
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 10)

    if ms == 0:
        return f"{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}.{ms}"
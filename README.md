# BeatsMatching Studio

**基于人体节拍动作序列检测的音视频同步可视化软件**

> 南京大学计算机科学与技术系 本科毕业设计  
> 作者：黄一凡（221870199）  
> 指导教师：过洁

## 项目简介

BeatsMatching Studio 是一款桌面视频编辑系统，能够自动检测视频中人体运动的节拍帧，并将其与音频节拍进行智能匹配，实现基于运动节奏的变速视频剪辑。

### 核心功能

- **双通道运动节拍检测**：基于 ViTPose（2D 关节速度）和 HMR2（3D 角速度）的多尺度等权帧级投票节拍检测
- **三种匹配算法**：贪心最近邻、全匹配动态规划、子集动态规划（支持速度比约束与平滑惩罚）
- **4 层推理加速**：SharedYOLO 缓存 → 批量推理 (bs=8) → FP16 半精度 → 帧预取流水线
- **完整 GUI**：PySide6 图形界面，集成时间轴编辑、姿态可视化、SMPL 3D 渲染、变速预览与导出

### 技术栈

| 类别 | 选择 |
|------|------|
| 语言 | Python 3.10 |
| GUI 框架 | PySide6 + qt-material |
| 深度学习 | PyTorch 2.5.1 + CUDA 12.1 |
| 2D 姿态估计 | ViTPose (HuggingFace transformers) |
| 3D 人体重建 | HMR2 / 4D-Humans (SMPL) |
| 人体检测 | YOLOv8n (ultralytics) |
| 音频处理 | librosa |
| 3D 渲染 | pyrender + trimesh |

---

## 项目结构

```
BeatsMatching-Studio/
├── main.py                          # 应用程序入口
├── requirements.txt                 # Python 依赖
├── environment.yml                  # Conda 环境配置（推荐）
├── .gitignore
│
├── core/                            # 核心数据层
│   ├── algorithm_config.py          # 算法参数配置（融合策略、后处理等）
│   ├── commands.py                  # Qt Undo/Redo 命令
│   ├── data_model.py                # ProjectModel 中心数据模型
│   ├── project_serializer.py        # 项目 JSON 序列化/反序列化
│   ├── structures.py                # Keyframe、EaseType、EffectType 数据结构
│   └── time_mapper.py               # 关键帧变速映射（缓动曲线插值）
│
├── engines/                         # 业务引擎层
│   ├── algorithms.py                # 三种匹配算法实现（贪心/全DP/子集DP）
│   ├── algorithm_proxy.py           # 算法代理（快速/深度学习切换）
│   ├── audio_loader.py              # 音频加载与节拍提取（librosa）
│   ├── enhanced_pose_worker.py      # 增强姿态提取 Worker（异常检测+多尺度投票）
│   ├── exporter.py                  # 变速渲染 + VFX + 音频合成导出
│   ├── inference_accelerator.py     # 4 层推理加速模块
│   ├── model_preloader.py           # 后台模型预加载
│   ├── player_engine.py             # 播放引擎（变速映射 + 音视频同步）
│   ├── pose_worker.py               # 基础姿态提取 Worker
│   ├── smpl_renderer.py             # SMPL 3D 人体渲染（pyrender）
│   ├── vfx_engine.py                # 视觉动效引擎（模糊/抖动/发光等）
│   └── video_loader.py              # 视频帧异步加载
│
├── gui/                             # 图形界面层
│   ├── main_window.py               # 主窗口
│   ├── player_widget.py             # 视频播放器控件
│   ├── property_panel.py            # 关键帧属性面板
│   ├── algorithm_config_panel.py    # 算法配置面板
│   ├── pose_visualization_panel.py  # 姿态可视化面板（2D骨架 + 3D SMPL）
│   ├── styles.py                    # 深色主题样式系统
│   └── timeline/                    # 时间轴组件
│       ├── timeline_container.py    # 时间轴容器
│       ├── header.py                # 时间刻度尺
│       ├── playhead.py              # 播放头
│       ├── tracks.py                # 视频/音频轨道
│       └── time_scaler.py           # 缩放坐标转换器
│
├── pose_extraction/                 # 深度学习姿态提取模块
│   ├── pipeline.py                  # 统一提取管线入口
│   ├── weights_config.py            # 模型权重路径配置
│   ├── core/                        # 信号处理核心
│   │   ├── beat_detector.py         # 节拍检测（三策略 + 投票）
│   │   ├── data_types.py            # 统一数据类型定义
│   │   └── smoothing.py             # 关节平滑与梯度计算
│   ├── vitpose/                     # ViTPose 2D 姿态估计
│   │   ├── detector.py              # 单帧检测
│   │   └── video_processor.py       # 视频批量处理
│   ├── hmr2/                        # HMR2 3D 人体重建
│   │   ├── reconstructor.py         # 单帧 SMPL 重建
│   │   └── video_processor.py       # 视频批量处理
│   ├── scripts/                     # 工具脚本
│   │   ├── download_weights.py      # HMR2 权重下载
│   │   ├── download_weights_robust.py  # 断点续传版下载
│   │   ├── download_smpl.py         # SMPL 模型下载
│   │   ├── migrate_weights.py       # 权重迁移工具
│   │   ├── run_pipeline.py          # 管线运行脚本
│   │   ├── verify_env.py            # 环境验证
│   │   └── test_*.py                # 各模块测试脚本
│   └── tests/                       # 单元测试
│       ├── test_beat_detector.py
│       ├── test_data_types.py
│       └── test_smoothing.py
│
├── experiments/                     # 论文实验代码
│   ├── run_thesis_experiments.py    # 表 5-4/5-5/5-6 匹配算法实验
│   ├── benchmark_layered_accel.py   # 表 5-7 推理加速基准测试
│   ├── generate_thesis_figures.py   # 图 4-4/4-8/5-2~5-6 论文图表生成
│   ├── test_algorithm_pipeline.py   # 端到端算法管线测试
│   ├── test_inference_acceleration.py  # 推理加速单项测试
│   └── analysis_correlation.py      # ViTPose/HMR2 信号相关性分析
│
├── resources/                       # 静态资源
│   └── icon.jpg                     # 应用图标
├── visualizations/                  # 可视化示例
│   ├── vitpose_skeleton.gif         # ViTPose 骨架动画
│   ├── hmr2_body_model.gif          # HMR2 人体模型动画
│   └── hmr2_rotation.gif            # HMR2 旋转示例
├── utils/                           # 通用工具
│   └── helpers.py                   # 时间格式化等辅助函数
└── docs/
    └── thesis_figures/              # 论文图表（代码生成）
```

---

## 环境配置

### 方式一：Conda 环境（推荐）

```bash
# 1. 创建 conda 环境
conda env create -f environment.yml -n pose_unified

# 2. 激活环境
conda activate pose_unified

# 3. 安装 4D-Humans（HMR2 依赖，必须 --no-deps 避免冲突）
pip install git+https://github.com/shubham-goel/4D-Humans.git --no-deps

# 4. 安装 GUI 依赖
pip install PySide6 qt-material librosa sounddevice trimesh pyrender moviepy
```

### 方式二：pip 安装

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. 安装 PyTorch（CUDA 12.1）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 4D-Humans
pip install git+https://github.com/shubham-goel/4D-Humans.git --no-deps

# 4. 安装其余依赖
pip install -r requirements.txt
```

### Windows 兼容性补丁

安装 4D-Humans 后需手动修改以下文件（Windows 不支持 EGL 渲染）：

1. **`hmr2/utils/__init__.py`**：`pyrender.OffscreenRenderer` 初始化用 `try/except` 包裹
2. **`hmr2/configs/__init__.py`**：`$HOME` 环境变量 fallback 到 `$USERPROFILE`
3. **`hmr2/models/hmr2.py`**：`SkeletonRenderer` 初始化添加 `None` 降级

### 验证安装

```bash
conda activate pose_unified
python pose_extraction/scripts/verify_env.py
```

---

## 模型权重下载

所有权重文件需放置在项目根目录下的 `weights/` 文件夹：

```
weights/
├── hmr2/
│   └── checkpoints/
│       └── epoch=35-step=1000000.ckpt    # HMR2 模型权重 (~200MB)
├── smpl/
│   └── SMPL_NEUTRAL.pkl                  # SMPL 中性人体模型 (~10MB)
├── yolo/
│   └── yolov8n.pt                        # YOLOv8n 人体检测 (~6MB)
└── vitpose/                              # ViTPose HuggingFace 缓存目录
```

### 下载方式

```bash
conda activate pose_unified

# HMR2 权重（从 4D-Humans 官方下载，支持断点续传）
python pose_extraction/scripts/download_weights_robust.py

# SMPL 人体模型（从 HuggingFace camenduru/4D-Humans 下载）
python pose_extraction/scripts/download_smpl.py

# YOLOv8n（首次运行时自动下载，或手动下载）
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# 下载后将 yolov8n.pt 移动到 weights/yolo/ 目录

# ViTPose（首次运行时自动从 HuggingFace 下载缓存到 weights/vitpose/）
# 模型 ID: usyd-community/vitpose-base-simple
# 若网络受限，可手动下载后设置 HF_HOME=weights/vitpose/

# 权重迁移（若权重已下载到其他位置）
python pose_extraction/scripts/migrate_weights.py
```

### 验证权重

```bash
python pose_extraction/weights_config.py
# 输出 [OK] 表示权重文件就位
```

---

## 测试数据

系统支持任意包含人体运动的视频文件。推荐使用 **AIST++ 舞蹈数据集** 进行测试：

- **数据来源**：[AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/)
- **推荐格式**：MP4，30fps，720p/1080p
- **推荐片段**：Breaking、Jazz、Hip-hop 风格，10~30 秒

将测试视频放置在任意位置，通过 GUI "打开视频" 功能导入。

---

## 使用方法

### 启动应用

```bash
conda activate pose_unified
python main.py
```

### 基本工作流

1. **导入媒体**：通过菜单导入舞蹈视频和音频文件
2. **提取姿态**：点击"提取姿态"，系统自动运行 ViTPose + HMR2 双通道推理
3. **自动匹配**：选择匹配算法（推荐"子集 DP"），点击"自动匹配"
4. **预览调整**：在时间轴上预览变速效果，拖拽微调关键帧
5. **导出视频**：点击"导出"，生成变速 + 动效 + 音频合成的最终视频

---

## 论文实验复现

所有论文中的实验均可通过 `experiments/` 目录下的脚本复现。

### 表 5-4/5-5/5-6：匹配算法对比实验

```bash
cd experiments
python run_thesis_experiments.py
# 输出：test_output/thesis_experiments/experiment_results.json
# 包含三种算法（贪心/全匹配DP/子集DP）的偏差、速度比标准差、锚定率对比
# 以及 w_smooth、w_skip 参数敏感性实验数据
```

### 表 5-7：推理加速基准测试

```bash
cd experiments
python benchmark_layered_accel.py
# 需要 GPU 和测试视频
# 输出：test_output/thesis_experiments/accel_benchmark.json
# 测量 5 种加速配置下的 ViTPose/HMR2 延时和总吞吐量
```

### 论文图表生成

```bash
cd experiments
python generate_thesis_figures.py
# 输出 7 张图到 docs/thesis_figures/：
#   图 4-4: 多尺度等权帧级投票示意
#   图 4-8: 四种缓动曲线对比
#   图 5-2: 运动信号与节拍检测可视化
#   图 5-3: 子集 DP 匹配结果可视化
#   图 5-4: 速度比序列对比
#   图 5-5: 参数敏感性分析
#   图 5-6: 推理加速效果柱状图
```

### 端到端管线测试

```bash
# 算法管线测试（需要 GPU + 测试视频）
cd experiments
python test_algorithm_pipeline.py --video <视频路径> --mode joint

# 推理加速单项测试
python test_inference_acceleration.py <视频路径>

# pose_extraction 单元测试
cd ../pose_extraction
pytest tests/ -v
```

---

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10 64-bit | Windows 11 |
| Python | 3.10 | 3.10 |
| GPU | NVIDIA GPU (CUDA 11.8+) | RTX 3060+ (6GB+ VRAM) |
| 内存 | 8 GB | 16 GB |
| 硬盘 | 2 GB（代码+权重） | 5 GB（含测试数据） |

### 实测性能（RTX 3060 Laptop GPU）

| 指标 | 数值 |
|------|------|
| ViTPose 推理 | ~40 ms/帧 |
| HMR2 推理 | ~45 ms/帧 (FP16) |
| 端到端吞吐量 | 12.7 fps（全优化） |
| 加速比 | 1.69x（相对基线） |

---

## 许可证

本项目为南京大学本科毕业设计作品，仅供学术研究使用。

### 第三方依赖许可

- **ViTPose**: Apache-2.0 License
- **4D-Humans / HMR2**: BSD License
- **SMPL**: [SMPL License](https://smpl.is.tue.mpg.de/modellicense.html) (仅限研究)
- **YOLOv8**: AGPL-3.0 License
- **PySide6**: LGPL-3.0 License

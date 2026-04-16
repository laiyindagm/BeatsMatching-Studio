"""
SMPL 人体模型渲染器
使用 SMPL_NEUTRAL.pkl + pyrender (GPU) 实现离屏渲染
不依赖 smplx 包，直接实现 SMPL 前向传播
"""
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

# pyrender 可用性检测
try:
    import trimesh
    import pyrender
    _HAS_PYRENDER = True
except ImportError:
    _HAS_PYRENDER = False


class SMPLRenderer:
    """
    轻量级 SMPL 渲染器
    
    从 SMPL_NEUTRAL.pkl 加载模型参数，
    根据 pose (24x3 轴角) 和 shape (10,) 参数生成人体 mesh，
    使用 matplotlib 渲染为图像。
    """
    
    def __init__(self, smpl_model_path: str):
        with open(smpl_model_path, 'rb') as f:
            self.model = pickle.load(f, encoding='latin1')
        
        self.v_template = self.model['v_template'].astype(np.float32)   # (6890, 3)
        self.faces = self.model['f'].astype(np.int32)                   # (13776, 3)
        self.shapedirs = self.model['shapedirs'].astype(np.float32)     # (6890, 3, 10)
        self.posedirs = self.model['posedirs'].astype(np.float32)       # (6890, 3, 207)
        self.J_regressor = self.model['J_regressor'].toarray().astype(np.float32)  # (24, 6890)
        self.weights = self.model['weights'].astype(np.float32)         # (6890, 24)
        self.kintree_table = self.model['kintree_table'].astype(np.int32)  # (2, 24)
        
        # 预计算骨骼树
        self.parent = self.kintree_table[0].astype(np.int32)
        self.parent[0] = -1
        
        # pyrender 离屏渲染器（懒加载）
        self._pyrender_renderer = None
        self._pyrender_size = None
        
        # 预分配顶点颜色（避免每帧重建）
        self._vertex_colors = np.tile(
            np.array([38, 166, 154, 204], dtype=np.uint8),
            (len(self.v_template), 1),
        )
        
        # pyrender Scene 缓存
        self._cached_scene = None
        self._cached_mesh_node = None
        self._cached_cam_node = None
        self._cached_light_node = None
        self._cached_camera = None
    
    def forward(self, pose: np.ndarray, shape: np.ndarray = None) -> np.ndarray:
        """
        SMPL 前向传播（向量化版本）
        
        Args:
            pose: (24, 3) 轴角旋转参数
            shape: (10,) shape 参数（可选，默认为 0）
        
        Returns:
            vertices: (6890, 3) 顶点坐标
        """
        if shape is None:
            shape = np.zeros(10, dtype=np.float32)
        
        # 1. Shape blend shapes
        v_shaped = self.v_template + np.einsum('vci,i->vc', self.shapedirs, shape)
        
        # 2. Joint locations from shaped mesh
        joints = self.J_regressor @ v_shaped  # (24, 3)
        
        # 3. Pose blend shapes（向量化：批量轴角→旋转矩阵）
        pose_rotmats = R.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(24, 3, 3).astype(np.float32)
        
        # (R - I) 展平，排除根节点 → (207,)
        ident = np.eye(3, dtype=np.float32)
        pose_feature_207 = (pose_rotmats[1:] - ident).reshape(-1)  # (23*9=207,)
        
        v_posed = v_shaped + np.einsum('vci,i->vc', self.posedirs, pose_feature_207)
        
        # 4. 骨骼树前向传播（构建全局变换矩阵）
        G = np.zeros((24, 4, 4), dtype=np.float32)
        G[0, :3, :3] = pose_rotmats[0]
        G[0, :3, 3] = joints[0]
        G[0, 3, 3] = 1.0
        
        for i in range(1, 24):
            p = self.parent[i]
            G_local = np.eye(4, dtype=np.float32)
            G_local[:3, :3] = pose_rotmats[i]
            G_local[:3, 3] = joints[i] - joints[p]
            G[i] = G[p] @ G_local
        
        # 减去关节初始位置的影响
        # G_final[i] = G[i] * [[I, -j_i], [0, 1]]
        G[:, :3, 3] -= np.einsum('ijk,ik->ij', G[:, :3, :3], joints)
        
        # 5. LBS 蒙皮（全向量化）
        # weights: (6890, 24), G: (24, 4, 4)
        # T = sum_i(w_i * G_i) → (6890, 4, 4)
        T = np.einsum('vj,jab->vab', self.weights, G)
        
        # 齐次坐标 (6890, 4)
        v_homo = np.hstack([v_posed, np.ones((len(v_posed), 1), dtype=np.float32)])
        
        # 变换 (6890, 4) @ (6890, 4, 4)^T → (6890, 4)
        transformed = np.einsum('vab,vb->va', T, v_homo)
        
        return transformed[:, :3]
    
    def render_frame(
        self,
        pose: np.ndarray,
        shape: np.ndarray = None,
        size: tuple = (400, 400),
        azimuth: float = 45.0,
        elevation: float = 20.0,
        flip_y: bool = False,
        stabilize_root: bool = False,
    ) -> np.ndarray:
        """
        渲染单帧 SMPL 人体模型
        优先使用 pyrender (GPU, ~10ms)，回退到 matplotlib (~300ms)
        """
        if stabilize_root:
            pose = pose.copy()
            pose[0] = np.zeros(3, dtype=pose.dtype)
        
        vertices = self.forward(pose, shape)
        
        # HMR2 Y-down → Y-up（独立于 stabilize_root）
        if flip_y:
            vertices[:, 1] *= -1
        
        if _HAS_PYRENDER:
            try:
                return self._render_pyrender(vertices, size, azimuth, elevation)
            except Exception as e:
                print(f"[SMPLRenderer] pyrender failed ({e}), falling back to matplotlib")
        return self._render_matplotlib(vertices, size, azimuth, elevation)
    
    def _get_pyrender_renderer(self, size):
        """懒初始化 pyrender 离屏渲染器"""
        if self._pyrender_renderer is None or self._pyrender_size != size:
            if self._pyrender_renderer is not None:
                try:
                    self._pyrender_renderer.delete()
                except Exception:
                    pass
            import sys
            if sys.platform == 'linux':
                os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
            self._pyrender_renderer = pyrender.OffscreenRenderer(
                viewport_width=size[0], viewport_height=size[1],
                point_size=1.0
            )
            self._pyrender_size = size
            # 尺寸变更时清除 Scene 缓存
            self._cached_scene = None
        return self._pyrender_renderer
    
    def _render_pyrender(self, vertices, size, azimuth, elevation):
        """pyrender GPU 渲染（缓存 Scene / Camera / Light，仅更换 Mesh 节点）"""
        renderer = self._get_pyrender_renderer(size)
        
        # ── 创建新 mesh（复用 trimesh 结构，仅更新顶点）──
        if not hasattr(self, '_cached_trimesh') or self._cached_trimesh is None:
            self._cached_trimesh = trimesh.Trimesh(
                vertices=vertices, faces=self.faces, process=False
            )
            self._cached_trimesh.visual.vertex_colors = self._vertex_colors
        else:
            self._cached_trimesh.vertices = vertices
        
        py_mesh = pyrender.Mesh.from_trimesh(self._cached_trimesh, smooth=True)
        
        # ── 计算相机位姿 ──
        center = vertices.mean(axis=0)
        extent = (vertices.max(axis=0) - vertices.min(axis=0)).max()
        cam_dist = extent * 1.8
        
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        cam_x = center[0] + cam_dist * np.cos(el_rad) * np.sin(az_rad)
        cam_y = center[1] + cam_dist * np.sin(el_rad)
        cam_z = center[2] + cam_dist * np.cos(el_rad) * np.cos(az_rad)
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        forward = center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, np.array([0, 1, 0]))
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(forward, np.array([1, 0, 0]))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = cam_pos
        
        # 光源位姿
        light_pose = np.eye(4)
        light_pose[:3, :3] = R.from_euler('xy', [elevation, azimuth], degrees=True).as_matrix()
        
        # ── 复用或重建 Scene ──
        if self._cached_scene is None:
            # 首次：完整创建
            scene = pyrender.Scene(
                bg_color=np.array([30, 30, 30, 255], dtype=np.uint8),
                ambient_light=np.array([0.3, 0.3, 0.3]),
            )
            self._cached_mesh_node = scene.add(py_mesh)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            self._cached_light_node = scene.add(light, pose=light_pose)
            self._cached_camera = pyrender.PerspectiveCamera(yfov=np.pi / 6)
            self._cached_cam_node = scene.add(self._cached_camera, pose=cam_pose)
            self._cached_scene = scene
        else:
            scene = self._cached_scene
            # 替换 mesh 节点
            scene.remove_node(self._cached_mesh_node)
            self._cached_mesh_node = scene.add(py_mesh)
            # 更新相机和光源位姿
            scene.set_pose(self._cached_cam_node, cam_pose)
            scene.set_pose(self._cached_light_node, light_pose)
        
        color, _ = renderer.render(scene)
        return color
    
    def _render_matplotlib(self, vertices, size, azimuth, elevation):
        """matplotlib fallback 渲染"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        vertices = vertices[:, [0, 2, 1]]  # Y-up → Z-up
        
        fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        mesh_collection = Poly3DCollection(
            vertices[self.faces],
            alpha=0.8, facecolor='#26a69a', edgecolor='#1e8e82', linewidth=0.1,
        )
        ax.add_collection3d(mesh_collection)
        ax.view_init(elev=elevation, azim=azimuth)
        
        center = vertices.mean(axis=0)
        max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2 + 0.1
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_axis_off()
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        image = buf.reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        
        return image
    
    def render_gif(
        self,
        poses: np.ndarray,
        output_path: str,
        shape: np.ndarray = None,
        fps: int = 15,
        azimuth: float = 45.0,
        elevation: float = 20.0,
        size: tuple = (400, 400),
        rotate_view: bool = True,
        flip_y: bool = False,
        stabilize_root: bool = False,
    ):
        """
        渲染 SMPL 人体模型动画 GIF
        
        Args:
            poses: (T, 24, 3) 轴角旋转参数序列
            output_path: GIF 输出路径
            shape: (10,) shape 参数
            fps: GIF 帧率
            azimuth: 初始水平旋转角度
            elevation: 俯仰角度
            size: 输出图像尺寸
            rotate_view: 是否缓慢旋转视角
            stabilize_root: 稳定根关节，消除全局朝向漂移
        """
        from PIL import Image
        
        frames = []
        T = len(poses)
        
        for i in range(T):
            azim = azimuth + (360.0 * i / T if rotate_view else 0)
            img = self.render_frame(
                poses[i], shape, size=size,
                azimuth=azim, elevation=elevation,
                flip_y=flip_y,
                stabilize_root=stabilize_root,
            )
            frames.append(Image.fromarray(img))
        
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            print(f"[SMPL GIF] 保存到: {output_path}")
            return output_path
        
        return None

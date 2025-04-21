from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                            QTableWidgetItem, QLabel, QComboBox, QPushButton,
                            QGraphicsView, QGraphicsScene)
from PyQt5.QtGui import QColor, QBrush, QPen, QImage, QPixmap, qRgb
from PyQt5.QtCore import Qt, QSize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
from matplotlib.figure import Figure
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['mathtext.default'] = 'regular'  # 设置数学文本默认字体

class TableViewer(QWidget):
    """表格形式显示数据"""
    def __init__(self, npy_file):
        super().__init__()
        self.npy_file = npy_file
        self.data_limits = {
            'enabled': True,
            'max_rows': 500,
            'max_cols': 500
        }
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 添加数据信息标签
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: blue")
        layout.addWidget(self.info_label)
        
        # 添加表格
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        # 初始化数据显示
        self.update_view()
    
    def set_data_limits(self, limits):
        """设置表格数据的显示限制
        
        Args:
            limits (dict): 包含以下键的字典:
                - enabled (bool): 是否启用数据限制
                - max_rows (int): 最大显示行数
                - max_cols (int): 最大显示列数
        """
        self.data_limits = limits
        self.update_view()

    def update_view(self):
        """更新表格视图以显示当前数据"""
        if not self.npy_file:
            return
        
        data = self.npy_file.get_slice_for_display()
        
        # 检查是否应用了数据限制
        if "(已限制显示)" in self.npy_file.current_slice_info:
            self.info_label.setText(f"注意: {self.npy_file.current_slice_info}")
            self.info_label.show()
        else:
            self.info_label.hide()
        
        # 调整表格大小
        if data.ndim == 1:
            # 一维数组显示为单行
            self.table.setRowCount(1)
            self.table.setColumnCount(len(data))
            
            for i in range(len(data)):
                self.table.setItem(0, i, QTableWidgetItem(str(data[i])))
        elif data.ndim == 2:
            # 二维数组直接显示
            self.table.setRowCount(data.shape[0])
            self.table.setColumnCount(data.shape[1])
            
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self.table.setItem(i, j, QTableWidgetItem(str(data[i, j])))
        else:
            # 高维数组不应该出现在这里，因为已经通过切片将其降维
            print("警告: 收到了高维数组")
            
        # 调整列宽以适应内容
        self.table.resizeColumnsToContents()

class ImageViewer(QWidget):
    """以图像形式显示数据"""
    def __init__(self, npy_file):
        super().__init__()
        self.npy_file = npy_file
        self.colormap = 'viridis'  # 默认颜色映射
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 添加数据信息标签
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: blue")
        layout.addWidget(self.info_label)
        
        # 添加颜色映射选择
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("颜色映射:"))
        
        self.colormap_combo = QComboBox()
        for cmap_name in ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                        'gray', 'hot', 'cool', 'rainbow', 'jet']:
            self.colormap_combo.addItem(cmap_name)
        
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        colormap_layout.addWidget(self.colormap_combo)
        colormap_layout.addStretch()
        
        layout.addLayout(colormap_layout)
        
        # 添加图像视图
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(0)  # 关闭抗锯齿以保持像素细节
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        
        layout.addWidget(self.graphics_view)
        
        # 初始化数据显示
        self.update_view()
    
    def update_view(self):
        """更新图像视图以显示当前数据"""
        if not self.npy_file:
            return
        
        self.scene.clear()
        
        data = self.npy_file.get_slice_for_display()
        
        # 检查是否应用了数据限制
        if "已限制显示" in self.npy_file.current_slice_info:
            self.info_label.setText(f"注意: {self.npy_file.current_slice_info}")
            self.info_label.show()
        else:
            self.info_label.hide()
        
        if data.ndim not in [1, 2]:
            self.scene.addText("无法将此维度的数据可视化为图像")
            return
        
        # 处理一维数据 - 转换为二维
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 如果数据类型不是浮点型，尝试转换
        if not np.issubdtype(data.dtype, np.floating):
            try:
                data = data.astype(float)
            except:
                # 如果转换失败，尝试保持原样
                pass
        
        # 归一化数据到0-1范围
        try:
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            if min_val != max_val:  # 避免除以零
                normalized_data = (data - min_val) / (max_val - min_val)
            else:
                normalized_data = np.zeros_like(data)
        except:
            self.scene.addText("无法归一化数据进行可视化")
            return
        
        # 应用颜色映射
        cmap = cm.get_cmap(self.colormap)
        colored_data = cmap(normalized_data)
        
        # 转换为RGB图像
        rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
        
        # 创建QImage
        height, width = rgb_data.shape[:2]
        bytes_per_line = 3 * width
        image = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 将图像添加到场景中
        pixmap = QPixmap.fromImage(image)
        self.scene.addPixmap(pixmap)
        
        # 调整视图大小
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def on_colormap_changed(self, cmap_name):
        """当颜色映射改变时更新视图"""
        self.colormap = cmap_name
        self.update_view()
    
    def resizeEvent(self, event):
        """当部件调整大小时，重新适应视图"""
        super().resizeEvent(event)
        if hasattr(self, 'scene') and hasattr(self, 'graphics_view'):
            self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

def create_visualization(data, vis_type):
    """创建各种类型的数据可视化"""
    if vis_type == "grayscale":
        _create_grayscale_visualization(data)
    elif vis_type == "heatmap":
        _create_heatmap_visualization(data)
    elif vis_type == "point_cloud":
        _create_point_cloud_visualization(data)
    elif vis_type == "timeseries":
        _create_timeseries_visualization(data)
    elif vis_type == "graph":
        _create_graph_visualization(data)
    else:
        raise ValueError(f"不支持的可视化类型: {vis_type}")

def _create_grayscale_visualization(data):
    """创建灰度图可视化"""
    # 确保数据是2D的
    if data.ndim == 1:
        # 转为单行的2D数组
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        raise ValueError("灰度图只支持1D或2D数据")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='gray')
    plt.colorbar(label='值')
    plt.title('灰度图显示')
    plt.tight_layout()
    plt.show()

def _create_heatmap_visualization(data):
    """创建热力图可视化"""
    # 确保数据是2D的
    if data.ndim == 1:
        # 转为单行的2D数组
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        raise ValueError("热力图只支持1D或2D数据")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='值')
    plt.title('热力图显示')
    
    # 添加坐标值
    plt.xticks(range(data.shape[1]))
    plt.yticks(range(data.shape[0]))
    
    plt.tight_layout()
    plt.show()

def _create_point_cloud_visualization(data):
    """创建3D点云可视化"""
    # 确保数据可以解释为点坐标
    if data.ndim == 1 and len(data) >= 3:
        # 单个点
        x, y, z = data[0], data[1], data[2]
        points = np.array([[x, y, z]])
    elif data.ndim == 2 and data.shape[1] >= 3:
        # 每行是一个点
        points = data[:, :3]
    else:
        raise ValueError("3D点云需要至少3个值的1D数组或形状为(n,3+)的2D数组")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('3D点云显示')
    
    plt.tight_layout()
    plt.show()

def _create_timeseries_visualization(data):
    """创建时间序列可视化"""
    # 确保数据是1D的或可以平展为1D
    if data.ndim > 1:
        data = data.flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), data, 'b-', linewidth=1.5)
    plt.title('时间序列显示')
    plt.xlabel('时间点')
    plt.ylabel('值')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def _create_graph_visualization(data):
    """创建有向图可视化"""
    # 确保数据是方阵（邻接矩阵）
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError("图显示需要方形邻接矩阵")
    
    # 创建有向图
    G = nx.DiGraph(data)
    
    plt.figure(figsize=(10, 8))
    
    # 设置节点位置
    pos = nx.spring_layout(G)
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos, arrows=True)
    
    # 添加节点标签
    labels = {i: str(i) for i in range(data.shape[0])}
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    # 添加边权重标签（如果不为零）
    edge_labels = {(i, j): f"{data[i, j]:.2f}" 
                  for i, j in G.edges() 
                  if abs(data[i, j]) > 1e-10}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('有向图显示')
    plt.axis('off')  # 关闭坐标轴
    
    plt.tight_layout()
    plt.show()
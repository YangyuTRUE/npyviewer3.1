from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QSpinBox, QPushButton, QSlider,
                           QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np

class DimensionControlWidget(QWidget):
    """高维数据切片控制面板"""
    
    # 当维度选择或索引改变时发出的信号
    dimension_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.npy_file = None
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # 创建显示维度选择组
        display_group = QGroupBox("选择要显示的维度")
        display_layout = QGridLayout()
        
        self.dim1_label = QLabel("行维度:")
        self.dim1_combo = QComboBox()
        self.dim1_combo.currentIndexChanged.connect(self.on_display_dim_changed)
        
        self.dim2_label = QLabel("列维度:")
        self.dim2_combo = QComboBox()
        self.dim2_combo.currentIndexChanged.connect(self.on_display_dim_changed)
        
        display_layout.addWidget(self.dim1_label, 0, 0)
        display_layout.addWidget(self.dim1_combo, 0, 1)
        display_layout.addWidget(self.dim2_label, 1, 0)
        display_layout.addWidget(self.dim2_combo, 1, 1)
        
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)
        
        # 创建帧控制组
        self.frame_group = QGroupBox("帧控制")
        self.frame_layout = QVBoxLayout()
        
        # 这里将动态添加控制不同维度的滑块
        self.frame_controls = {}  # 将存储每个维度的控制部件
        
        self.frame_group.setLayout(self.frame_layout)
        main_layout.addWidget(self.frame_group)
        
        # 添加动画控制按钮
        anim_group = QGroupBox("动画控制")
        anim_layout = QHBoxLayout()
        
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_animation)
        self.play_button.setEnabled(False)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_animation)
        self.stop_button.setEnabled(False)
        
        self.anim_dim_combo = QComboBox()
        self.anim_dim_combo.setPlaceholderText("选择动画维度")
        self.anim_dim_combo.currentIndexChanged.connect(self.on_anim_dim_changed)
        
        anim_layout.addWidget(QLabel("动画维度:"))
        anim_layout.addWidget(self.anim_dim_combo)
        anim_layout.addWidget(self.play_button)
        anim_layout.addWidget(self.stop_button)
        
        anim_group.setLayout(anim_layout)
        main_layout.addWidget(anim_group)
        
        # 添加恢复按钮
        self.reset_button = QPushButton("恢复原始数据")
        self.reset_button.clicked.connect(self.reset_to_original)
        main_layout.addWidget(self.reset_button)
        
        # 添加弹性空间
        main_layout.addStretch()
        
        # 动画计时器
        self.animation_timer = None
        self.animating_dimension = None
    
    def setup(self, npy_file):
        """设置控制器以处理特定的NPY文件"""
        self.npy_file = npy_file
        data = npy_file.get_current_data()
        ndim = data.ndim
        
        # 在操作之前阻断信号，避免初始化过程中的错误触发
        self.blockSignals(True)
        
        # 重置控件
        self.dim1_combo.clear()
        self.dim2_combo.clear()
        self.anim_dim_combo.clear()
        
        # 清除现有的帧控制
        for widget in self.frame_controls.values():
            self.frame_layout.removeWidget(widget['container'])
            widget['container'].deleteLater()
        self.frame_controls = {}
        
        # 如果数据不是高维的，禁用控制
        if ndim <= 2:
            self.setEnabled(False)
            self.blockSignals(False)  # 恢复信号处理
            return
            
        self.setEnabled(True)
        
        # 填充维度组合框
        for i in range(ndim):
            dim_name = f"维度 {i} (大小: {data.shape[i]})"
            self.dim1_combo.addItem(dim_name, i)
            self.dim2_combo.addItem(dim_name, i)
            self.anim_dim_combo.addItem(dim_name, i)
        
        # 创建每个维度的帧控制
        for i in range(ndim):
            self._create_frame_control(i, data.shape[i])
        
        # 默认选择前两个维度作为显示维度
        if ndim > 1:
            self.dim1_combo.setCurrentIndex(0)
            self.dim2_combo.setCurrentIndex(1)
        
        # 恢复信号处理
        self.blockSignals(False)
        
        # 初始化显示
        self.on_display_dim_changed()
    
    def _create_frame_control(self, dim_index, dim_size):
        """为特定维度创建帧控制滑块"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(f"维度 {dim_index}:")
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, dim_size - 1)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(max(1, dim_size // 10))
        
        spinbox = QSpinBox()
        spinbox.setRange(0, dim_size - 1)
        spinbox.setValue(0)
        
        # 连接控件
        slider.valueChanged.connect(lambda v: spinbox.setValue(v))
        spinbox.valueChanged.connect(lambda v: slider.setValue(v))
        spinbox.valueChanged.connect(lambda v, d=dim_index: self.on_frame_changed(d, v))
        
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        
        self.frame_layout.addWidget(container)
        
        self.frame_controls[dim_index] = {
            'container': container,
            'slider': slider,
            'spinbox': spinbox,
            'label': label
        }
    
    def on_display_dim_changed(self):
        """当显示维度改变时更新UI"""
        if not self.npy_file:
            return
        
        # 获取选中的维度
        dim1 = self.dim1_combo.currentData()
        dim2 = self.dim2_combo.currentData()
        
        # 如果任何一个维度为None，使用默认值
        if dim1 is None or dim2 is None:
            data_ndim = self.npy_file.data.ndim
            available_dims = list(range(data_ndim))
            
            if dim1 is None:
                dim1 = 0 if available_dims else None
            
            if dim2 is None:
                if len(available_dims) > 1:
                    dim2 = 1
                elif available_dims:
                    dim2 = 0
                else:
                    dim2 = None
        
        # 继续检查是否选择了相同维度
        if dim1 is not None and dim2 is not None and dim1 == dim2:
            # 如果选择了相同的维度，尝试选择不同的维度
            available_dims = list(range(self.npy_file.data.ndim))
            if dim1 in available_dims:
                available_dims.remove(dim1)
            
            if available_dims:
                if dim1 == self.dim1_combo.currentData():
                    self.dim2_combo.setCurrentIndex(self.dim2_combo.findData(available_dims[0]))
                else:
                    self.dim1_combo.setCurrentIndex(self.dim1_combo.findData(available_dims[0]))
                return
        
        # 确保维度值有效后再设置
        if dim1 is not None and dim2 is not None:
            try:
                self.npy_file.set_display_dimensions((dim1, dim2))
                
                # 更新帧控制的可见性
                for dim, controls in self.frame_controls.items():
                    if dim in (dim1, dim2):
                        controls['container'].hide()
                    else:
                        controls['container'].show()
                
                # 发出维度改变信号
                self.dimension_changed.emit()
            except Exception as e:
                print(f"设置显示维度出错: {str(e)}")
    
    def on_frame_changed(self, dimension, value):
        """当帧控制改变时更新数据视图"""
        if not self.npy_file:
            return
        
        try:
            self.npy_file.set_dimension_index(dimension, value)
            self.dimension_changed.emit()
        except Exception as e:
            print(f"设置维度索引出错: {str(e)}")
    
    def on_anim_dim_changed(self):
        """当选择的动画维度改变时"""
        dim = self.anim_dim_combo.currentData()
        if dim is not None:
            self.animating_dimension = dim
            self.play_button.setEnabled(True)
    
    def toggle_animation(self):
        """开始/暂停动画"""
        import time
        from PyQt5.QtCore import QTimer
        
        if not self.animation_timer:
            # 开始动画
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.advance_frame)
            self.animation_timer.start(100)  # 每100毫秒更新一次
            self.play_button.setText("暂停")
            self.stop_button.setEnabled(True)
        else:
            # 暂停动画
            self.animation_timer.stop()
            self.animation_timer = None
            self.play_button.setText("播放")
    
    def advance_frame(self):
        """前进到下一帧"""
        if not self.npy_file or self.animating_dimension is None:
            return
        
        dim = self.animating_dimension
        controls = self.frame_controls.get(dim)
        if not controls:
            return
        
        # 获取当前值和最大值
        current = controls['spinbox'].value()
        maximum = controls['spinbox'].maximum()
        
        # 计算下一个值（循环）
        next_value = (current + 1) % (maximum + 1)
        
        # 更新控件值（这会触发on_frame_changed）
        controls['spinbox'].setValue(next_value)
    
    def stop_animation(self):
        """停止动画并重置"""
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None
        
        self.play_button.setText("播放")
        self.stop_button.setEnabled(False)
    
    def reset_to_original(self):
        """恢复原始数据视图"""
        if self.npy_file:
            self.npy_file.restore_original_data()
            self.setup(self.npy_file)  # 重新设置控件
            self.dimension_changed.emit()
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog,
                            QTabWidget, QTextEdit, QFormLayout, QGroupBox, QSpinBox,
                            QDoubleSpinBox, QDateEdit, QCheckBox, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QDate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import argparse
import json
import re
from datetime import datetime, timedelta
# 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class NumpyArrayGenerator:
    def __init__(self):
        self.methods = {
            'random': self.generate_random,
            'zeros': self.generate_zeros,
            'ones': self.generate_ones,
            'arange': self.generate_arange,
            'linspace': self.generate_linspace,
            'identity': self.generate_identity,
            'nested': self.generate_nested,
            'time_series': self.generate_time_series,
            'graph': self.generate_graph,
            'heightmap': self.generate_heightmap,
            'custom': self.generate_custom
        }

    def generate_array(self, method, shape, **kwargs):
        """根据指定的方法生成numpy数组"""
        if method not in self.methods:
            raise ValueError(f"不支持的方法: {method}。可用方法: {', '.join(self.methods.keys())}")
        
        return self.methods[method](shape, **kwargs)
    
    def generate_random(self, shape, distribution='uniform', **kwargs):
        """生成随机数组"""
        shape = self._parse_shape(shape)
        
        if distribution == 'uniform':
            low = float(kwargs.get('low', 0.0))
            high = float(kwargs.get('high', 1.0))
            return np.random.uniform(low, high, shape)
        elif distribution == 'normal':
            mean = float(kwargs.get('mean', 0.0))
            std = float(kwargs.get('std', 1.0))
            return np.random.normal(mean, std, shape)
        elif distribution == 'poisson':
            lam = float(kwargs.get('lam', 1.0))
            return np.random.poisson(lam, shape)
        elif distribution == 'binomial':
            n = int(kwargs.get('n', 10))
            p = float(kwargs.get('p', 0.5))
            return np.random.binomial(n, p, shape)
        else:
            raise ValueError(f"不支持的分布: {distribution}")
    
    def generate_zeros(self, shape, **kwargs):
        """生成全零数组"""
        shape = self._parse_shape(shape)
        return np.zeros(shape)
    
    def generate_ones(self, shape, **kwargs):
        """生成全一数组"""
        shape = self._parse_shape(shape)
        return np.ones(shape)
    
    def generate_arange(self, shape, **kwargs):
        """生成等差数列"""
        start = float(kwargs.get('start', 0))
        stop = float(kwargs.get('stop', 10))
        step = float(kwargs.get('step', 1))
        return np.arange(start, stop, step)
    
    def generate_linspace(self, shape, **kwargs):
        """生成均分数列"""
        start = float(kwargs.get('start', 0))
        stop = float(kwargs.get('stop', 10))
        num = int(kwargs.get('num', 50))
        return np.linspace(start, stop, num)
    
    def generate_identity(self, shape, **kwargs):
        """生成单位矩阵"""
        n = int(kwargs.get('n', 3))
        return np.identity(n)
    
    def generate_nested(self, shape, **kwargs):
        """生成嵌套数组"""
        if 'data' in kwargs:
            data_str = kwargs['data']
            # 尝试将字符串转换为Python对象
            try:
                # 尝试使用JSON解析
                data = json.loads(data_str)
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试使用eval（注意安全风险）
                try:
                    data = eval(data_str)
                except Exception as e:
                    raise ValueError(f"无法解析嵌套数据: {e}")
            
            return np.array(data)
        else:
            raise ValueError("需要提供'data'参数来生成嵌套数组")
    
    def generate_time_series(self, shape, **kwargs):
        """生成时间序列数据"""
        length = int(kwargs.get('length', 100))
        start_date = kwargs.get('start_date', datetime.now().strftime('%Y-%m-%d'))
        frequency = kwargs.get('frequency', 'daily')
        trend = kwargs.get('trend', 'linear')
        
        # 解析开始日期
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # 生成日期序列
        dates = []
        if frequency == 'daily':
            dates = [start_date + timedelta(days=i) for i in range(length)]
        elif frequency == 'hourly':
            dates = [start_date + timedelta(hours=i) for i in range(length)]
        elif frequency == 'weekly':
            dates = [start_date + timedelta(weeks=i) for i in range(length)]
        elif frequency == 'monthly':
            dates = [start_date + timedelta(days=30*i) for i in range(length)]
        
        # 生成值序列
        values = np.zeros(length)
        if trend == 'linear':
            slope = float(kwargs.get('slope', 0.1))
            intercept = float(kwargs.get('intercept', 0))
            values = intercept + slope * np.arange(length)
        elif trend == 'exponential':
            base = float(kwargs.get('base', 1.1))
            values = np.power(base, np.arange(length))
        elif trend == 'seasonal':
            period = int(kwargs.get('period', 7))
            amplitude = float(kwargs.get('amplitude', 1.0))
            values = amplitude * np.sin(2 * np.pi * np.arange(length) / period)
        
        # 添加随机波动
        noise_level = float(kwargs.get('noise', 0.1))
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, length)
            values += noise
        
        # 返回结果
        if kwargs.get('return_dates', False):
            return np.array([(d.strftime('%Y-%m-%d'), v) for d, v in zip(dates, values)], 
                            dtype=[('date', 'U10'), ('value', 'float64')])
        else:
            return values
    
    def generate_graph(self, shape, **kwargs):
        """生成图结构数据"""
        n_nodes = int(kwargs.get('n_nodes', 10))
        edge_probability = float(kwargs.get('edge_probability', 0.3))
        directed = kwargs.get('directed', False)
        weighted = kwargs.get('weighted', False)
        
        # 生成邻接矩阵
        adj_matrix = np.random.random((n_nodes, n_nodes)) < edge_probability
        
        # 对角线设为0（没有自环）
        np.fill_diagonal(adj_matrix, 0)
        
        # 如果是无向图，则确保对称性
        if not directed:
            adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T
        
        # 如果是加权图，则用随机值替换True
        if weighted:
            weights = np.random.uniform(0, 1, (n_nodes, n_nodes))
            adj_matrix = adj_matrix.astype(float) * weights
        
        return adj_matrix
    
    def generate_heightmap(self, shape, **kwargs):
        """生成高度图数据"""
        shape = self._parse_shape(shape)
        if len(shape) != 2:
            shape = (256, 256) if not shape else (shape[0], shape[0])
        
        method = kwargs.get('method', 'perlin')
        
        if method == 'perlin':
            # 简单的Perlin噪声模拟
            x = np.linspace(0, 5, shape[0])
            y = np.linspace(0, 5, shape[1])
            x_grid, y_grid = np.meshgrid(x, y)
            
            # 生成几个不同频率的噪声
            noise1 = np.sin(x_grid) * np.cos(y_grid)
            noise2 = np.sin(2*x_grid) * np.cos(2*y_grid) * 0.5
            noise3 = np.sin(4*x_grid) * np.cos(4*y_grid) * 0.25
            
            heightmap = noise1 + noise2 + noise3
            return heightmap
        
        elif method == 'random':
            # 随机高度图
            heightmap = np.random.random(shape)
            return heightmap
        
        elif method == 'gaussian':
            # 高斯混合模型
            centers = int(kwargs.get('centers', 5))
            heightmap = np.zeros(shape)
            
            for _ in range(centers):
                cx = np.random.randint(0, shape[0])
                cy = np.random.randint(0, shape[1])
                sigma = np.random.uniform(shape[0]/10, shape[0]/5)
                amplitude = np.random.uniform(0.5, 1.0)
                
                x = np.arange(shape[0])
                y = np.arange(shape[1])
                x_grid, y_grid = np.meshgrid(x, y)
                
                gaussian = amplitude * np.exp(-((x_grid-cx)**2 + (y_grid-cy)**2) / (2*sigma**2))
                heightmap += gaussian
            
            return heightmap
        
        else:
            raise ValueError(f"不支持的高度图方法: {method}")
    
    def generate_custom(self, shape, **kwargs):
        """生成自定义数组"""
        if 'function' in kwargs:
            func_str = kwargs['function']
            shape = self._parse_shape(shape)
            
            # 创建一个带有numpy的局部环境
            local_env = {'np': np, 'shape': shape}
            
            # 执行函数字符串
            try:
                # 注意：这里存在安全风险，生产环境中应该使用更安全的方法
                exec(f"result = {func_str}", {}, local_env)
                return local_env['result']
            except Exception as e:
                raise ValueError(f"自定义函数执行失败: {e}")
        else:
            raise ValueError("需要提供'function'参数来生成自定义数组")
    
    def _parse_shape(self, shape):
        """解析形状参数"""
        if isinstance(shape, (list, tuple)):
            return shape
        
        if isinstance(shape, int):
            return (shape,)
        
        if isinstance(shape, str):
            # 尝试将字符串转换为元组
            try:
                # 处理可能的形式如 '(3, 3)' 或 '3, 3'
                shape_str = shape.strip()
                if shape_str.startswith('(') and shape_str.endswith(')'):
                    shape_str = shape_str[1:-1]
                
                # 分割字符串并转换为整数
                shape_parts = [int(part.strip()) for part in shape_str.split(',') if part.strip()]
                if not shape_parts:  # 如果解析后为空列表
                    return (1,)  # 返回默认形状
                return tuple(shape_parts)
            except Exception as e:
                print(f"形状解析错误: {e}, 使用默认形状 (1,)")
                return (1,)  # 解析失败时返回默认形状
        
        print(f"不支持的形状参数类型: {type(shape)}, 使用默认形状 (1,)")
        return (1,)  # 不支持的类型时返回默认形状
    
class ArrayVisualizer(FigureCanvas):
    """用于可视化NumPy数组的Matplotlib画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(ArrayVisualizer, self).__init__(self.fig)
        self.setParent(parent)
    
    def plot_array(self, arr):
        """根据数组维度绘制可视化"""
        # 完全清除图形
        self.fig.clear()
        # 重新添加子图
        self.axes = self.fig.add_subplot(111)
        
        if arr is None:
            return
            
        if len(arr.shape) == 1:
            # 1D数组：线图
            self.axes.plot(arr)
            self.axes.set_title("一维数组")
        elif len(arr.shape) == 2:
            # 2D数组：热图
            if arr.shape[0] <= 100 and arr.shape[1] <= 100:
                im = self.axes.imshow(arr, cmap='viridis')
                self.fig.colorbar(im, ax=self.axes)
                self.axes.set_title("二维数组热图")
            else:
                self.axes.text(0.5, 0.5, f"数组太大，无法可视化\n形状: {arr.shape}", 
                              horizontalalignment='center', verticalalignment='center')
        else:
            # 高维数组：显示形状信息
            self.axes.text(0.5, 0.5, f"高维数组，无法可视化\n形状: {arr.shape}", 
                          horizontalalignment='center', verticalalignment='center')
        
        self.fig.tight_layout()
        self.draw()

class NPYGeneratorGUI(QMainWindow):
    """NumPy数组生成器GUI界面"""
    def __init__(self):
        super(NPYGeneratorGUI, self).__init__()
        self.generator = NumpyArrayGenerator()
        self.current_array = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('NumPy数组生成器')
        self.setGeometry(100, 100, 1000, 600)
        
        # 创建中心窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 方法选择
        method_group = QGroupBox("生成方法")
        method_layout = QVBoxLayout(method_group)
        self.method_combo = QComboBox()
        self.method_combo.addItems(self.generator.methods.keys())
        self.method_combo.currentIndexChanged.connect(self.update_param_fields)
        method_layout.addWidget(self.method_combo)
        control_layout.addWidget(method_group)
        
        # 形状参数
        shape_group = QGroupBox("数组形状")
        shape_layout = QFormLayout(shape_group)
        self.shape_input = QLineEdit("(3, 3)")
        shape_layout.addRow("形状:", self.shape_input)
        control_layout.addWidget(shape_group)
        
        # 方法特定参数
        self.param_group = QGroupBox("方法参数")
        self.param_layout = QFormLayout(self.param_group)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.param_group)
        scroll_area.setWidgetResizable(True)
        control_layout.addWidget(scroll_area)
        
        # 操作按钮
        button_group = QGroupBox("操作")
        button_layout = QVBoxLayout(button_group)
        
        self.generate_button = QPushButton("生成数组")
        self.generate_button.clicked.connect(self.generate_array)
        button_layout.addWidget(self.generate_button)
        
        self.save_button = QPushButton("保存到文件")
        self.save_button.clicked.connect(self.save_array)
        self.save_button.setEnabled(False)  # 初始禁用
        button_layout.addWidget(self.save_button)
        
        control_layout.addWidget(button_group)
        control_layout.addStretch()
        
        # 右侧显示区域
        display_panel = QTabWidget()
        
        # 可视化标签页
        visual_tab = QWidget()
        visual_layout = QVBoxLayout(visual_tab)
        self.visualizer = ArrayVisualizer(self, width=5, height=4)
        visual_layout.addWidget(self.visualizer)
        display_panel.addTab(visual_tab, "可视化")
        
        # 数据标签页
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        data_layout.addWidget(self.data_text)
        display_panel.addTab(data_tab, "数据")
        
        # 信息标签页
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        display_panel.addTab(info_tab, "信息")
        
        # 添加面板到主布局
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(display_panel, 2)
        
        # 初始更新参数字段
        self.update_param_fields()
    
    def update_param_fields(self):
        """根据选择的方法更新参数输入字段"""
        # 清除现有的参数字段
        while self.param_layout.rowCount() > 0:
            self.param_layout.removeRow(0)
        
        method = self.method_combo.currentText()
        
        # 根据选择的方法添加相应的参数字段
        if method == "random":
            self.distributions = QComboBox()
            self.distributions.addItems(["uniform", "normal", "poisson", "binomial"])
            self.param_layout.addRow("分布:", self.distributions)
            
            self.low = QDoubleSpinBox()
            self.low.setRange(-1000, 1000)
            self.low.setValue(0.0)
            self.param_layout.addRow("最小值 (low):", self.low)
            
            self.high = QDoubleSpinBox()
            self.high.setRange(-1000, 1000)
            self.high.setValue(1.0)
            self.param_layout.addRow("最大值 (high):", self.high)
            
            self.mean = QDoubleSpinBox()
            self.mean.setRange(-1000, 1000)
            self.mean.setValue(0.0)
            self.param_layout.addRow("均值 (mean):", self.mean)
            
            self.std = QDoubleSpinBox()
            self.std.setRange(0, 1000)
            self.std.setValue(1.0)
            self.param_layout.addRow("标准差 (std):", self.std)
            
            self.lam = QDoubleSpinBox()
            self.lam.setRange(0, 1000)
            self.lam.setValue(1.0)
            self.param_layout.addRow("lambda:", self.lam)
            
            self.n = QSpinBox()
            self.n.setRange(1, 1000)
            self.n.setValue(10)
            self.param_layout.addRow("试验次数 (n):", self.n)
            
            self.p = QDoubleSpinBox()
            self.p.setRange(0, 1)
            self.p.setValue(0.5)
            self.p.setSingleStep(0.1)
            self.param_layout.addRow("成功概率 (p):", self.p)
        
        elif method in ["zeros", "ones"]:
            # 这两种方法只需要形状参数，不需要额外参数
            pass
        
        elif method == "arange":
            self.start = QDoubleSpinBox()
            self.start.setRange(-1000, 1000)
            self.start.setValue(0)
            self.param_layout.addRow("起始值:", self.start)
            
            self.stop = QDoubleSpinBox()
            self.stop.setRange(-1000, 1000)
            self.stop.setValue(10)
            self.param_layout.addRow("结束值:", self.stop)
            
            self.step = QDoubleSpinBox()
            self.step.setRange(0.01, 100)
            self.step.setValue(1)
            self.param_layout.addRow("步长:", self.step)
        
        elif method == "linspace":
            self.start = QDoubleSpinBox()
            self.start.setRange(-1000, 1000)
            self.start.setValue(0)
            self.param_layout.addRow("起始值:", self.start)
            
            self.stop = QDoubleSpinBox()
            self.stop.setRange(-1000, 1000)
            self.stop.setValue(10)
            self.param_layout.addRow("结束值:", self.stop)
            
            self.num = QSpinBox()
            self.num.setRange(2, 1000)
            self.num.setValue(50)
            self.param_layout.addRow("点数:", self.num)
        
        elif method == "identity":
            self.n = QSpinBox()
            self.n.setRange(1, 100)
            self.n.setValue(3)
            self.param_layout.addRow("矩阵大小:", self.n)
        
        elif method == "nested":
            self.data = QTextEdit()
            self.data.setPlaceholderText("输入嵌套数组数据，例如: [[1, 2], [3, 4]]")
            self.data.setMaximumHeight(100)
            self.param_layout.addRow("数据:", self.data)
        
        elif method == "time_series":
            self.length = QSpinBox()
            self.length.setRange(2, 1000)
            self.length.setValue(100)
            self.param_layout.addRow("序列长度:", self.length)
            
            self.start_date = QDateEdit()
            self.start_date.setDate(QDate.currentDate())
            self.param_layout.addRow("开始日期:", self.start_date)
            
            self.frequency = QComboBox()
            self.frequency.addItems(["daily", "hourly", "weekly", "monthly"])
            self.param_layout.addRow("频率:", self.frequency)
            
            self.trend = QComboBox()
            self.trend.addItems(["linear", "exponential", "seasonal"])
            self.trend.currentIndexChanged.connect(self.update_trend_params)
            self.param_layout.addRow("趋势:", self.trend)
            
            self.slope = QDoubleSpinBox()
            self.slope.setRange(-100, 100)
            self.slope.setValue(0.1)
            self.param_layout.addRow("斜率:", self.slope)
            
            self.intercept = QDoubleSpinBox()
            self.intercept.setRange(-100, 100)
            self.intercept.setValue(0)
            self.param_layout.addRow("截距:", self.intercept)
            
            self.base = QDoubleSpinBox()
            self.base.setRange(0.01, 10)
            self.base.setValue(1.1)
            self.base.setEnabled(False)
            self.param_layout.addRow("指数基数:", self.base)
            
            self.period = QSpinBox()
            self.period.setRange(2, 100)
            self.period.setValue(7)
            self.period.setEnabled(False)
            self.param_layout.addRow("周期:", self.period)
            
            self.amplitude = QDoubleSpinBox()
            self.amplitude.setRange(0.01, 100)
            self.amplitude.setValue(1.0)
            self.amplitude.setEnabled(False)
            self.param_layout.addRow("振幅:", self.amplitude)
            
            self.noise = QDoubleSpinBox()
            self.noise.setRange(0, 10)
            self.noise.setValue(0.1)
            self.param_layout.addRow("噪声级别:", self.noise)
            
            self.return_dates = QCheckBox("包含日期")
            self.param_layout.addRow("", self.return_dates)
        
        elif method == "graph":
            self.n_nodes = QSpinBox()
            self.n_nodes.setRange(2, 100)
            self.n_nodes.setValue(10)
            self.param_layout.addRow("节点数:", self.n_nodes)
            
            self.edge_probability = QDoubleSpinBox()
            self.edge_probability.setRange(0, 1)
            self.edge_probability.setValue(0.3)
            self.edge_probability.setSingleStep(0.1)
            self.param_layout.addRow("边概率:", self.edge_probability)
            
            self.directed = QCheckBox("有向图")
            self.param_layout.addRow("", self.directed)
            
            self.weighted = QCheckBox("加权图")
            self.param_layout.addRow("", self.weighted)
        
        elif method == "heightmap":
            self.heightmap_method = QComboBox()
            self.heightmap_method.addItems(["perlin", "random", "gaussian"])
            self.param_layout.addRow("高度图方法:", self.heightmap_method)
            
            self.centers = QSpinBox()
            self.centers.setRange(1, 20)
            self.centers.setValue(5)
            self.param_layout.addRow("高斯中心数:", self.centers)
        
        elif method == "custom":
            self.function = QTextEdit()
            self.function.setPlaceholderText("输入NumPy函数，例如: np.sin(np.linspace(0, 2*np.pi, shape[0]))")
            self.function.setMaximumHeight(100)
            self.param_layout.addRow("自定义函数:", self.function)
    
    def update_trend_params(self):
        """根据选择的时间序列趋势更新相应参数的可用性"""
        trend = self.trend.currentText()
        
        # 设置线性趋势参数
        self.slope.setEnabled(trend == "linear")
        self.intercept.setEnabled(trend == "linear")
        
        # 设置指数趋势参数
        self.base.setEnabled(trend == "exponential")
        
        # 设置季节性趋势参数
        self.period.setEnabled(trend == "seasonal")
        self.amplitude.setEnabled(trend == "seasonal")
    
    def generate_array(self):
        """根据用户输入生成NumPy数组"""
        try:
            method = self.method_combo.currentText()
            shape = self.shape_input.text()
            
            params = {}
            
            # 根据不同方法收集参数
            if method == "random":
                distribution = self.distributions.currentText()
                params['distribution'] = distribution
                
                if distribution == "uniform":
                    params['low'] = self.low.value()
                    params['high'] = self.high.value()
                elif distribution == "normal":
                    params['mean'] = self.mean.value()
                    params['std'] = self.std.value()
                elif distribution == "poisson":
                    params['lam'] = self.lam.value()
                elif distribution == "binomial":
                    params['n'] = self.n.value()
                    params['p'] = self.p.value()
            
            elif method == "arange":
                params['start'] = self.start.value()
                params['stop'] = self.stop.value()
                params['step'] = self.step.value()
            
            elif method == "linspace":
                params['start'] = self.start.value()
                params['stop'] = self.stop.value()
                params['num'] = self.num.value()
            
            elif method == "identity":
                params['n'] = self.n.value()
            
            elif method == "nested":
                params['data'] = self.data.toPlainText()
            
            elif method == "time_series":
                params['length'] = self.length.value()
                # 将QDate转换为字符串，避免传递QDate对象
                params['start_date'] = self.start_date.date().toString("yyyy-MM-dd")
                params['frequency'] = self.frequency.currentText()
                params['trend'] = self.trend.currentText()
                params['slope'] = self.slope.value()
                params['intercept'] = self.intercept.value()
                params['base'] = self.base.value()
                params['period'] = self.period.value()
                params['amplitude'] = self.amplitude.value()
                params['noise'] = self.noise.value()
                params['return_dates'] = self.return_dates.isChecked()
            
            elif method == "graph":
                params['n_nodes'] = self.n_nodes.value()
                params['edge_probability'] = self.edge_probability.value()
                params['directed'] = self.directed.isChecked()
                params['weighted'] = self.weighted.isChecked()
            
            elif method == "heightmap":
                params['method'] = self.heightmap_method.currentText()
                if self.heightmap_method.currentText() == "gaussian":
                    params['centers'] = self.centers.value()
            
            elif method == "custom":
                params['function'] = self.function.toPlainText()
            
            # 生成数组 - 修复此处调用，确保正确传递shape参数
            self.current_array = self.generator.generate_array(method, shape, **params)
            
            # 更新显示
            self.update_array_display()
            
            # 启用保存按钮
            self.save_button.setEnabled(True)
            self.update()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成数组时出错：{str(e)}")
            # 打印更详细的错误信息，便于调试
            import traceback
            traceback.print_exc()
    
    def update_array_display(self):
        """更新数组显示"""
        if self.current_array is None:
            return
        
        # 更新可视化
        self.visualizer.plot_array(self.current_array)
        
        # 更新数据文本
        if self.current_array.size <= 1000:  # 限制显示大小
            self.data_text.setText(str(self.current_array))
        else:
            self.data_text.setText(f"数组太大，无法完全显示。\n前100个元素:\n{str(self.current_array.flat[:100])}")
        
        # 更新信息文本
        info = f"形状: {self.current_array.shape}\n"
        info += f"类型: {self.current_array.dtype}\n"
        info += f"大小: {self.current_array.size}\n"
        info += f"维度: {self.current_array.ndim}\n"
        info += f"内存用量: {self.current_array.nbytes / 1024:.2f} KB\n"
        
        if self.current_array.size > 0:
            try:
                info += f"最小值: {self.current_array.min()}\n"
                info += f"最大值: {self.current_array.max()}\n"
                info += f"均值: {self.current_array.mean()}\n"
                info += f"标准差: {self.current_array.std()}\n"
            except TypeError:
                # 处理可能包含非数值数据的数组
                pass
        
        self.info_text.setText(info)
        
        # 强制刷新界面
        QApplication.processEvents()
    
    def save_array(self):
        """保存数组到文件"""
        if self.current_array is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "保存NumPy数组", "", "NumPy Files (*.npy);;All Files (*)")
        if filename:
            if not filename.endswith(".npy"):
                filename += ".npy"
            try:
                np.save(filename, self.current_array)
                QMessageBox.information(self, "保存成功", f"数组已保存到：{filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存文件时出错：{str(e)}")

def main():
    app = QApplication(sys.argv)
    gui = NPYGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

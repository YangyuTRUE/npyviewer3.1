#!/usr/bin/env python3
import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QTableWidget, 
                             QAction, QFileDialog, QMessageBox, QDockWidget)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from pathlib import Path

from data_handler import NPYFile, load_file, save_file
from slice_controls import DimensionControlWidget
from visualizers import TableViewer, ImageViewer, create_visualization
from data_limit_dialog import DataLimitDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.npy_file = None
        self.current_viewer = None
        
        # 初始化数据限制设置
        self.data_limits = {
            'enabled': True,
            'max_rows': 500,
            'max_cols': 500
        }
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("NPYViewer 3.1")
        self.setGeometry(100, 100, 1000, 700)
        
        # 创建中央部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 信息标签
        self.info_label = QLabel("NPY Viewer 3.1 - 请打开NPY文件")
        self.main_layout.addWidget(self.info_label)
        
        # 数据显示区域
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.main_layout.addWidget(self.viewer_container)
        
        # 创建菜单和工具栏
        self.create_menu()
        
        # 创建维度控制面板（初始隐藏）
        self.dimension_dock = QDockWidget("维度控制", self)
        self.dimension_dock.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.dimension_control = DimensionControlWidget(self)
        self.dimension_dock.setWidget(self.dimension_control)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dimension_dock)
        self.dimension_dock.hide()
        
        self.show()
    
    def create_menu(self):
        # 文件菜单
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        
        open_action = QAction("打开", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        table_view_action = QAction("表格视图", self)
        table_view_action.triggered.connect(lambda: self.switch_view("table"))
        view_menu.addAction(table_view_action)
        
        image_view_action = QAction("图像视图", self)
        image_view_action.triggered.connect(lambda: self.switch_view("image"))
        view_menu.addAction(image_view_action)
        
        # 可视化菜单
        vis_menu = menubar.addMenu("可视化")
        
        grayscale_action = QAction("灰度图", self)
        grayscale_action.triggered.connect(lambda: self.visualize("grayscale"))
        vis_menu.addAction(grayscale_action)
        
        heatmap_action = QAction("热力图", self)
        heatmap_action.triggered.connect(lambda: self.visualize("heatmap"))
        vis_menu.addAction(heatmap_action)
        
        point_cloud_action = QAction("3D点云", self)
        point_cloud_action.triggered.connect(lambda: self.visualize("point_cloud"))
        vis_menu.addAction(point_cloud_action)
        
        timeseries_action = QAction("时间序列", self)
        timeseries_action.triggered.connect(lambda: self.visualize("timeseries"))
        vis_menu.addAction(timeseries_action)
        
        graph_action = QAction("有向图", self)
        graph_action.triggered.connect(lambda: self.visualize("graph"))
        vis_menu.addAction(graph_action)
        
        # 添加设置菜单
        settings_menu = menubar.addMenu("设置")
        
        data_limit_action = QAction("数据显示限制", self)
        data_limit_action.triggered.connect(self.show_data_limit_dialog)
        settings_menu.addAction(data_limit_action)
    
    def open_file(self):
        # 修改默认目录为程序所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开NPY文件", current_dir,
            "NumPy Files (*.npy);;CSV Files (*.csv);;All Files (*.*)")
        
        if file_path:
            try:
                data, file_type = load_file(file_path)
                self.npy_file = NPYFile(data, file_path)
                
                # 检查数据尺寸，如果任意维度大于500，自动启用数据限制
                if any(dim > 500 for dim in data.shape):
                    self.data_limits['enabled'] = True
                    self.npy_file.set_data_limits(self.data_limits)
                    # 显示提示信息
                    QMessageBox.information(self, "数据限制已启用", 
                                          "由于数据尺寸较大（任意维度大于500），已自动启用数据显示限制。\n"
                                          f"显示限制为：最大行数 {self.data_limits['max_rows']}，最大列数 {self.data_limits['max_cols']}。\n")
                                        #   "您可以在"设置"菜单中修改此设置。")
                else:
                    self.npy_file.set_data_limits(self.data_limits)
                
                self.update_ui()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开文件: {str(e)}")

    def save_file(self):
        if not self.npy_file:
            QMessageBox.warning(self, "警告", "没有数据可保存")
            return
        
        # 保持一致性，保存文件对话框也默认在程序所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存文件", current_dir,
            "NumPy Files (*.npy);;CSV Files (*.csv);;MAT Files (*.mat);;All Files (*.*)")
        
        if file_path:
            try:
                save_file(self.npy_file.get_current_data(), file_path)
                QMessageBox.information(self, "成功", "文件保存成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存文件失败: {str(e)}")
    
    def show_data_limit_dialog(self):
        """显示数据限制设置对话框"""
        dialog = DataLimitDialog(self, self.data_limits)
        if dialog.exec_():
            # 如果用户确认了设置，更新数据限制
            self.data_limits = dialog.get_limits()
            
            # 如果已经加载了数据，应用新的限制设置
            if self.npy_file:
                self.npy_file.set_data_limits(self.data_limits)
                # 更新当前视图
                if self.current_viewer:
                    if hasattr(self.current_viewer, 'set_data_limits'):
                        self.current_viewer.set_data_limits(self.data_limits)
                    self.current_viewer.update_view()
    
    def update_ui(self):
        """更新UI以显示当前加载的NPY文件信息"""
        if self.npy_file:
            self.info_label.setText(str(self.npy_file))
            
            # 根据维度决定是否显示维度控制面板
            data = self.npy_file.get_current_data()
            if data.ndim > 2:
                self.dimension_control.setup(self.npy_file)
                self.dimension_dock.show()
            else:
                self.dimension_dock.hide()
            
            # 默认使用表格视图
            self.switch_view("table")
    
    def switch_view(self, view_type):
        """切换视图类型（表格/图像）"""
        if not self.npy_file:
            return
        
        # 清除当前视图
        self.clear_current_view()
        
        # 创建新视图
        if view_type == "table":
            self.current_viewer = TableViewer(self.npy_file)
            if hasattr(self.current_viewer, 'set_data_limits'):
                self.current_viewer.set_data_limits(self.data_limits)
        elif view_type == "image":
            self.current_viewer = ImageViewer(self.npy_file)
            if hasattr(self.current_viewer, 'set_data_limits'):
                self.current_viewer.set_data_limits(self.data_limits)
        
        # 将视图添加到布局中
        self.viewer_layout.addWidget(self.current_viewer)
        
        # 连接维度控制器信号到当前视图
        if hasattr(self.dimension_control, 'dimension_changed'):
            self.dimension_control.dimension_changed.connect(self.current_viewer.update_view)
    
    def clear_current_view(self):
        """清除当前视图"""
        if self.current_viewer:
            self.viewer_layout.removeWidget(self.current_viewer)
            self.current_viewer.deleteLater()
            self.current_viewer = None
    
    def visualize(self, vis_type):
        """创建特定类型的可视化"""
        if not self.npy_file:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        
        try:
            create_visualization(self.npy_file.get_current_data(), vis_type)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"无法创建可视化: {str(e)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格获得更现代的外观
    window = MainWindow()
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            try:
                data, file_type = load_file(file_path)
                window.npy_file = NPYFile(data, file_path)
                
                # 检查数据尺寸，如果任意维度大于500，自动启用数据限制
                if any(dim > 500 for dim in data.shape):
                    window.data_limits['enabled'] = True
                    window.npy_file.set_data_limits(window.data_limits)
                else:
                    window.npy_file.set_data_limits(window.data_limits)
                    
                window.update_ui()
            except Exception as e:
                print(f"无法打开文件: {str(e)}")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
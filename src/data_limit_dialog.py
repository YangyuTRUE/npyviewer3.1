from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QSpinBox, QCheckBox, QDialogButtonBox, QGroupBox)
from PyQt5.QtCore import Qt

class DataLimitDialog(QDialog):
    """设置数据显示限制的对话框"""
    def __init__(self, parent=None, current_limits=None):
        super().__init__(parent)
        self.setWindowTitle("数据范围限制设置")
        self.setMinimumWidth(350)
        
        # 初始化默认值
        if current_limits is None:
            current_limits = {
                'enabled': True,
                'max_rows': 500,
                'max_cols': 500
            }
        self.current_limits = current_limits
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 启用限制选项
        self.enable_limit_cb = QCheckBox("启用数据显示限制")
        self.enable_limit_cb.setChecked(self.current_limits['enabled'])
        self.enable_limit_cb.toggled.connect(self.on_enable_toggled)
        layout.addWidget(self.enable_limit_cb)
        
        # 限制设置组
        limit_group = QGroupBox("数据显示限制")
        limit_layout = QVBoxLayout()
        
        # 行限制
        row_limit_layout = QHBoxLayout()
        row_limit_layout.addWidget(QLabel("最大行数:"))
        self.row_limit_spin = QSpinBox()
        self.row_limit_spin.setRange(10, 100000)
        self.row_limit_spin.setValue(self.current_limits['max_rows'])
        self.row_limit_spin.setSingleStep(100)
        row_limit_layout.addWidget(self.row_limit_spin)
        limit_layout.addLayout(row_limit_layout)
        
        # 列限制
        col_limit_layout = QHBoxLayout()
        col_limit_layout.addWidget(QLabel("最大列数:"))
        self.col_limit_spin = QSpinBox()
        self.col_limit_spin.setRange(10, 100000)
        self.col_limit_spin.setValue(self.current_limits['max_cols'])
        self.col_limit_spin.setSingleStep(100)
        col_limit_layout.addWidget(self.col_limit_spin)
        limit_layout.addLayout(col_limit_layout)
        
        # 警告提示
        warning_label = QLabel("注意: 过大的数据可能会导致程序响应缓慢或崩溃")
        warning_label.setStyleSheet("color: red")
        limit_layout.addWidget(warning_label)
        
        limit_group.setLayout(limit_layout)
        layout.addWidget(limit_group)
        
        # 对话框按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # 更新UI状态
        self.on_enable_toggled(self.enable_limit_cb.isChecked())
        
    def on_enable_toggled(self, enabled):
        """当启用/禁用限制时更新UI状态"""
        self.row_limit_spin.setEnabled(enabled)
        self.col_limit_spin.setEnabled(enabled)
        
    def get_limits(self):
        """返回用户设置的限制"""
        return {
            'enabled': self.enable_limit_cb.isChecked(),
            'max_rows': self.row_limit_spin.value(),
            'max_cols': self.col_limit_spin.value()
        }
import numpy as np
import pandas as pd
from scipy.io import savemat
import os

class NPYFile:
    def __init__(self, data, filename):
        self.data = data
        self.filename = filename
        self.original_data = data.copy()  # 保存原始数据副本
        self.current_slice_info = "完整数据"
        
        # 当前维度选择（用于高维数据）
        self.display_dims = (0, 1) if data.ndim > 1 else (0,)
        self.current_indices = [0] * data.ndim  # 每个维度的当前索引
        
        # 数据限制设置
        self.data_limits = {
            'enabled': True,
            'max_rows': 500,
            'max_cols': 500
        }
    
    def __str__(self):
        """返回文件信息的字符串表示"""
        info = [
            f"文件名: {os.path.basename(self.filename)}",
            f"数据类型: {self.data.dtype}",
            f"形状: {self.data.shape}",
            f"维度: {self.data.ndim}",
            f"当前视图: {self.current_slice_info}"
        ]
        return "\n".join(info)
    
    def get_current_data(self):
        """获取当前正在查看的数据"""
        return self.data
    
    def set_data_limits(self, limits):
        """设置数据显示限制
        
        Args:
            limits (dict): 包含以下键的字典:
                - enabled (bool): 是否启用数据限制
                - max_rows (int): 最大显示行数
                - max_cols (int): 最大显示列数
        """
        self.data_limits = limits
    
    def set_display_dimensions(self, dims):
        """设置要显示的维度"""
        # 增加对None的检查
        if dims is None or any(d is None for d in dims):
            # 使用默认值处理None情况
            if self.data.ndim > 1:
                dims = (0, 1)  # 使用前两个维度
            else:
                dims = (0,)  # 一维数据只有一个维度
        
        if len(dims) > 2 or any(d >= self.data.ndim for d in dims):
            raise ValueError("无效的显示维度")
        
        self.display_dims = dims
        return self.get_slice_for_display()
    
    def set_dimension_index(self, dim, index):
        """设置特定维度的当前索引值"""
        if dim < 0 or dim >= self.data.ndim:
            raise ValueError(f"维度 {dim} 超出范围")
        
        if index < 0 or index >= self.data.shape[dim]:
            raise ValueError(f"索引 {index} 超出维度 {dim} 的范围")
        
        self.current_indices[dim] = index
        return self.get_slice_for_display()
    
    def get_slice_for_display(self):
        """根据当前显示维度和索引获取要显示的切片"""
        if not self.data.size:
            return np.array([])
            
        # 创建用于切片的索引列表
        indices = []
        for i in range(self.data.ndim):
            if i in self.display_dims:
                indices.append(slice(None))  # 使用完整的维度
            else:
                indices.append(self.current_indices[i])  # 使用特定索引
        
        # 应用切片
        sliced_data = self.data[tuple(indices)]
        
        # 应用数据限制
        limited_data, is_limited = self._apply_data_limits(sliced_data)
        
        # 更新切片信息
        self.current_slice_info = self._get_slice_info()
        if is_limited:
            self.current_slice_info += " (已限制显示)"
        
        return limited_data
    
    def _apply_data_limits(self, data):
        """应用数据限制，返回可能被裁剪的数据和是否被限制的标志"""
        if not self.data_limits['enabled']:
            return data, False
            
        # 检查是否需要限制
        is_limited = False
        limited_data = data
        
        # 处理一维数据
        if data.ndim == 1:
            if len(data) > self.data_limits['max_cols']:
                limited_data = data[:self.data_limits['max_cols']]
                is_limited = True
                
        # 处理二维数据
        elif data.ndim == 2:
            rows, cols = data.shape
            row_limit = min(rows, self.data_limits['max_rows'])
            col_limit = min(cols, self.data_limits['max_cols'])
            
            if row_limit < rows or col_limit < cols:
                limited_data = data[:row_limit, :col_limit]
                is_limited = True
                
        return limited_data, is_limited
    
    def _get_slice_info(self):
        """生成当前切片的描述信息"""
        info = []
        for i in range(self.data.ndim):
            if i in self.display_dims:
                info.append(f"维度{i}:完整")
            else:
                info.append(f"维度{i}:索引{self.current_indices[i]}")
        return ", ".join(info)
    
    def restore_original_data(self):
        """恢复到原始数据"""
        self.data = self.original_data.copy()
        self.current_slice_info = "完整数据"
        self.current_indices = [0] * self.data.ndim
        return self.data

def load_file(file_path):
    """加载NPY或CSV文件"""
    if file_path.lower().endswith('.npy'):
        data = np.load(file_path, allow_pickle=True)
        file_type = 'npy'
    elif file_path.lower().endswith('.csv'):
        data = np.array(pd.read_csv(file_path).values.tolist())
        file_type = 'csv'
    else:
        raise ValueError("不支持的文件类型，请使用.npy或.csv文件")
    
    return data, file_type

def save_file(data, file_path):
    """保存数据到指定路径"""
    if file_path.lower().endswith('.npy'):
        np.save(file_path, data)
    elif file_path.lower().endswith('.csv'):
        np.savetxt(file_path, data, delimiter=',')
    elif file_path.lower().endswith('.mat'):
        savemat(file_path, {'data': data})
    else:
        # 默认保存为NPY
        if not file_path.lower().endswith('.npy'):
            file_path += '.npy'
        np.save(file_path, data)
# npyviewer2.0


# numpy_creator

## 项目简介
`numpy_creator` 是一个用于创建和操作NumPy数组的Python可视化程序。该程序提供了一些便捷的方法来生成、修改和分析NumPy数组。

## 安装说明
1. 克隆此仓库到本地：
    ```bash
    git clone https://github.com/YangyuTRUE/npyviewer2.0.git
    ```
2. 进入项目目录：
    ```bash
    cd npyviewer2.0
    ```
3. 安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法
1. 导入`numpy_creator`模块：
    ```python
    import numpy_creator as nc
    ```
2. 创建一个NumPy数组：
    ```python
    array = nc.create_array((3, 3), fill_value=0)
    print(array)
    ```
3. 修改数组：
    ```python
    modified_array = nc.modify_array(array, new_value=1, position=(0, 0))
    print(modified_array)
    ```

## 示例
以下是一个完整的示例：
```python
python numpy_creator 
```

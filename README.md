# 双目/多相机标定工具

基于特征匹配的双相机标定与坐标转换工具。

## 功能

- 双视角图像标定（特征匹配法）
- 像素坐标 → 实际坐标转换
- 交互式精度验证（点击图像获取坐标）

## 环境依赖

```bash
pip install opencv-python numpy matplotlib
```

## 使用方法

### 1. 准备图像

将两张图像放入 `data/` 目录：
- `wall_camera.jpg` - 墙装相机视角
- `robot_camera.jpg` - 扫地机视角

### 2. 运行标定

```bash
python src/calibration.py --image1 data/wall_camera.jpg --image2 data/robot_camera.jpg
```

### 3. 交互式验证

标定完成后，会显示交互式界面：
- 在墙装相机图像上点击任意位置
- 终端会输出该点的实际坐标（单位：米）

### 4. 测试示例

```bash
python src/demo.py
```

## 项目结构

```
camera-calibration/
├── data/                    # 图像数据目录
│   ├── wall_camera.jpg     # 墙装相机图像
│   └── robot_camera.jpg    # 扫地机图像
├── src/                    # 源代码
│   ├── __init__.py
│   ├── calibration.py       # 标定主程序
│   ├── coordinate.py        # 坐标转换
│   ├── visualization.py     # 可视化
│   └── demo.py            # 示例
├── models/                 # 模型参数存储
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖
└── README.md
```

## 核心算法

### 特征匹配标定

1. ORB 特征提取
2. BFMatcher 匹配
3. RANSAC + 本质矩阵过滤
4. decompose(E) 获取 R, t

### 坐标转换

```python
# 像素 → 世界坐标
X, Y, Z = pixel_to_world(u, v, R, t, K)
```

## 参数说明

| 参数 | 说明 |
|------|------|
| R | 旋转矩阵 (3x3) |
| t | 平移向量 (3x1) |
| K | 相机内参矩阵 |
| u, v | 像素坐标 |

## 示例输出

```
=== 标定完成 ===
旋转矩阵 R:
[[ 0.98 -0.15  0.12]
 [ 0.14  0.99  0.05]
 [-0.11 -0.04  0.99]]
平移向量 t: [0.5, 0.1, 2.0] 米

=== 坐标转换 ===
点击像素坐标 (320, 240) -> 实际坐标 (1.23, 0.85, 0.00) 米
```

## 注意事项

- 确保两图像有足够的特征重叠区域
- 建议采集 15-20 组图像对
- 特征丰富的环境效果更好

---

*Author: fenghuohuo2001*
*License: MIT*
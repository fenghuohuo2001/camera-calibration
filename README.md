# 双目/多相机标定工具

基于特征匹配的双相机标定与坐标转换工具。

## 功能

- 双视角图像标定（特征匹配法）
- 像素坐标 → 实际坐标转换
- 交互式精度验证（点击图像获取坐标）

## 环境依赖

```bash
pip install opencv-python numpy matplotlib pyyaml
```

## 使用方法

### 1. 克隆仓库

```bash
git clone https://github.com/fenghuohuo2001/camera-calibration.git
cd camera-calibration
git pull origin main
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备图像

将两张图像放入 `data/` 目录：
- `20251203-205714.jpg` - 墙装相机视角（默认）
- `20251203-210012.jpg` - 扫地机视角（默认）

或使用自己的图像，通过 `--image1` 和 `--image2` 指定。

### 4. 运行标定 + 交互验证

```bash
python src/calibration.py --calibrate --interactive
```

参数说明：
- `--calibrate` - 执行标定（可选）
- `--interactive` - 交互式验证模式
- `--image1 <path>` - 墙装相机图像路径
- `--image2 <path>` - 扫地机图像图像
- `--output <path>` - 输出参数文件（默认 models/calibration.yaml）

### 5. 单独交互验证

如果已有标定参数文件：

```bash
python src/calibration.py --interactive
```

### 6. 运行演示

```bash
python src/demo.py
```

## 交互式验证操作

1. 运行标定后会自动打开图像窗口
2. **点击图像任意位置** - 在图像上标记并显示坐标
3. **按 'q' 或 'ESC'** - 退出

## 项目结构

```
camera-calibration/
├── data/                    # 图像数据目录
│   ├── 20251203-205714.jpg  # 墙装相机图像
│   └── 20251203-210012.jpg  # 扫地机图像
├── src/                     # 源代码
│   ├── __init__.py
│   ├── calibration.py       # 标定主程序
│   ├── coordinate.py        # 坐标转换
│   ├── visualization.py     # 可视化
│   └── demo.py              # 示例
├── models/                  # 模型参数存储
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖
└── README.md
```

## 核心算法

### 特征匹配标定

1. ORB 特征提取
2. BFMatcher 匹配 + 比率测试
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

## 1080P 相机内参（已内置）

```python
K = [[1340.45, 0, 957.66],
     [0, 1338.90, 514.79],
     [0, 0, 1]]

dist = [-0.459, 0.264, 0.0006, 0.0006, -0.083]
```

## 常见问题

### Q: 标定结果不准确？
A: 检查特征匹配数量（需 >50），确保两图像有足够的公共视野区域。

### Q: 点击坐标变化大但实际坐标不变？
A: 可能是标定参数不正确，尝试重新标定或检查匹配点质量。

### Q: 图像太大显示不下？
A: 程序会自动缩小到 1/2 显示，点击坐标会自动还原。

---

*Author: fenghuohuo2001*
*License: MIT*
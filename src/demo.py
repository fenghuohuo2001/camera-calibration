# -*- coding: utf-8 -*-
"""
演示程序 - 展示标定和坐标转换功能
"""

import numpy as np
from src.calibration import CameraCalibrator
from src.coordinate import pixel_to_world_simple, compute_distance


def create_test_images():
    """创建测试图像（纯色+特征点）"""
    import cv2
    
    # 创建测试图像
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加一些特征点
    np.random.seed(42)
    for _ in range(100):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        cv2.circle(img1, (x, y), 3, (255, 255, 255), -1)
    
    # 第二张图像加一些偏移
    for _ in range(100):
        x = np.random.randint(50, 590) + 20
        y = np.random.randint(50, 430) + 10
        cv2.circle(img2, (x, y), 3, (255, 255, 255), -1)
    
    # 保存测试图像
    cv2.imwrite('data/wall_camera.jpg', img1)
    cv2.imwrite('data/robot_camera.jpg', img2)
    
    print("测试图像已创建")


def demo_coordinate_conversion():
    """演示坐标转换"""
    print("\n=== 坐标转换演示 ===")
    
    # 模拟参数
    fx, fy = 800, 800
    cx, cy = 320, 240
    camera_height = 2.5  # 相机高度2.5米
    
    # 测试点
    test_points = [
        (320, 240),   # 图像中心
        (100, 100),   # 左上
        (540, 380),   # 右下
    ]
    
    for u, v in test_points:
        coord = pixel_to_world_simple(u, v, fx, fy, cx, cy, camera_height)
        if coord:
            print(f"像素 ({u}, {v}) -> 世界 ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}) 米")
    
    # 计算两点距离
    p1 = (1.0, 1.0, 0)
    p2 = (2.0, 1.0, 0)
    dist = compute_distance(p1, p2)
    print(f"\n两点距离: {dist:.2f} 米")


def demo_with_loaded_params():
    """演示使用已加载参数的坐标转换"""
    print("\n=== 使用标定参数演示 ===")
    
    # 创建标定器
    calibrator = CameraCalibrator()
    
    # 模拟标定结果
    calibrator.R = np.array([
        [0.98, -0.15, 0.12],
        [0.14, 0.99, 0.05],
        [-0.11, -0.04, 0.99]
    ])
    calibrator.t = np.array([0.5, 0.1, 2.0]).reshape(3, 1)
    
    # 测试坐标转换
    test_points = [(100, 100), (320, 240), (500, 400)]
    
    for u, v in test_points:
        coord = calibrator.pixel_to_world(u, v)
        if coord:
            print(f"像素 ({u}, {v}) -> 实际坐标: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) 米")


def main():
    print("=" * 50)
    print("双相机标定工具 - 演示程序")
    print("=" * 50)
    
    # 1. 创建测试图像
    print("\n[1] 创建测试图像...")
    create_test_images()
    
    # 2. 坐标转换演示
    print("\n[2] 坐标转换演示...")
    demo_coordinate_conversion()
    
    # 3. 使用标定参数演示
    print("\n[3] 使用标定参数演示...")
    demo_with_loaded_params()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("使用方法:")
    print("  python src/calibration.py --image1 data/wall_camera.jpg \\")
    print("                              --image2 data/robot_camera.jpg \\")
    print("                              --interactive")
    print("=" * 50)


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
双相机标定主程序
功能：特征匹配标定 + 坐标转换 + 交互式验证
"""

print(">>> 正在导入模块...")
import cv2
print(f">>> cv2 版本: {cv2.__version__}")
import numpy as np
print(">>> numpy 导入成功")
import argparse
import os
import sys
import yaml
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(">>> 准备导入 visualization...")
try:
    from visualization import draw_matches
    print(">>> visualization 导入成功")
except Exception as e:
    print(f">>> visualization 导入失败: {e}")
    draw_matches = None


class CameraCalibrator:
    """双相机标定类"""
    
    # 1080P相机内参
    DEFAULT_K = np.array([
        [1340.4545644883922, 0.0, 957.6642584789628],
        [0.0, 1338.9037588649903, 514.7896498420388],
        [0.0, 0.0, 1.0]
    ])
    
    # 畸变系数
    DEFAULT_DIST = np.array([-0.4592577581462052, 0.26447244392183217, 
                            0.0005528469178028297, 0.0005887615350833584, 
                            -0.0831977348681754])
    
    def __init__(self, config=None):
        self.R = None  # 旋转矩阵
        self.t = None  # 平移向量
        self.K_wall = self.DEFAULT_K  # 墙装相机内参
        self.K_robot = self.DEFAULT_K  # 扫地机内参
        self.dist = self.DEFAULT_DIST  # 畸变系数
        self.feature_matches = []
        self.scale_factor = 1.0  # 尺度因子（米/归一化单位）
        
        # 加载配置文件（可选）
        if config:
            self.load_config(config)
        
    def set_scale(self, pixel_distance, real_distance_cm):
        """
        设置尺度校准
        
        Args:
            pixel_distance: 两个像素点之间的像素距离
            real_distance_cm: 实际距离（厘米）
        """
        if pixel_distance > 0:
            self.scale_factor = real_distance_cm / 100.0 / pixel_distance
            print(f">>> 尺度已校准: 1 像素 = {self.scale_factor:.4f} 米")
        
    def calibrate(self, image1_path, image2_path):
        """
        执行双相机标定
        
        Args:
            image1_path: 墙装相机图像路径
            image2_path: 扫地机图像路径
            
        Returns:
            R, t: 旋转矩阵和平移向量
        """
        # 读取图像
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("无法读取图像，请检查路径")
        
        print(f"图像1尺寸: {img1.shape}")
        print(f"图像2尺寸: {img2.shape}")
        
        # 畸变校正
        h, w = img1.shape[:2]
        new_k, roi = cv2.getOptimalNewCameraMatrix(self.K_wall, self.dist, (w, h), 1, (w, h))
        img1_undistorted = cv2.undistort(img1, self.K_wall, self.dist, None, new_k)
        img2_undistorted = cv2.undistort(img2, self.K_wall, self.dist, None, new_k)
        
        # 使用校正后的图像进行特征提取
        img1 = img1_undistorted
        img2 = img2_undistorted
        
        print("已完成畸变校正")
        
        # 特征提取 - 使用SIFT（更高精度）
        sift = cv2.SIFT_create(nfeatures=2000)
        
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        print(f"图像1特征点: {len(kp1)}")
        print(f"图像2特征点: {len(kp2)}")
        
        # 特征匹配 - SIFT使用L2距离
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 比率测试过滤
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        print(f"好的匹配点: {len(good_matches)}")
        
        if len(good_matches) < 10:
            raise ValueError("匹配点太少，请检查图像是否有足够重叠区域")
        
        # 提取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # 计算本质矩阵 + RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            method=cv2.RANSAC, 
            prob=0.99, 
            threshold=3.0
        )
        
        # 分解本质矩阵获取R, t
        _, R, t, mask = cv2.recoverPose(
            E, pts1, pts2,
            focal=1.0,  # 归一化焦距
            pp=(0, 0)   # 主点
        )
        
        self.R = R
        self.t = t
        self.feature_matches = good_matches
        self.kp1 = kp1
        self.kp2 = kp2
        
        # 保存特征匹配可视化结果
        result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, 'match_result.jpg')
        
        # 读取原始图像用于可视化
        img1_orig = cv2.imread(image1_path)
        img2_orig = cv2.imread(image2_path)
        
        draw_matches(img1_orig, img2_orig, kp1, kp2, good_matches, 
                     output_path=output_path, max_matches=50, scale=0.3)
        
        return R, t
    
    def pixel_to_world(self, u, v, K=None, Z=0):
        """
        像素坐标转世界坐标
        
        Args:
            u, v: 像素坐标
            K: 相机内参矩阵（可选，默认使用1080P相机内参）
            Z: 目标点高度（默认地面=0）
            
        Returns:
            (X, Y, Z): 世界坐标（米）
        """
        if self.R is None or self.t is None:
            raise ValueError("请先执行标定")
        
        # 使用默认内参
        if K is None:
            K = self.K_wall
        
        # 像素 → 归一化平面
        x_norm = (u - K[0, 2]) / K[0, 0]
        y_norm = (v - K[1, 2]) / K[1, 1]
        
        # 像素 → 归一化平面
        x_norm = (u - K[0, 2]) / K[0, 0]
        y_norm = (v - K[1, 2]) / K[1, 1]
        
        # 反向变换
        R_inv = self.R.T
        t_vec = self.t.flatten()  # 确保是一维向量
        t_inv = -self.R.T @ t_vec
        
        # 射线与地面交点计算 (Z=0)
        denom = R_inv[2, 0] * x_norm + R_inv[2, 1] * y_norm + R_inv[2, 2]
        
        if abs(denom) < 1e-6:
            return None
        
        d = -t_inv[2] / denom
        
        # 计算世界坐标（应用尺度因子）
        P_camera = d * np.array([x_norm, y_norm, 1])
        P_world = R_inv @ P_camera + t_inv
        
        # 应用尺度因子转换为米
        X = float(P_world[0] * self.scale_factor)
        Y = float(P_world[1] * self.scale_factor)
        Z = float(P_world[2] * self.scale_factor)
        
        return (X, Y, Z)
    
    def save_params(self, filepath):
        """保存标定参数"""
        if self.R is None or self.t is None:
            raise ValueError("没有可保存的参数")
        
        params = {
            'R': self.R.tolist(),
            't': self.t.tolist(),
            'K_wall': self.K_wall.tolist(),
            'K_robot': self.K_robot.tolist(),
            'distortion': self.dist.tolist()
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(params, f)
        
        print(f"参数已保存到: {filepath}")
    
    def load_params(self, filepath):
        """加载标定参数"""
        with open(filepath, 'r') as f:
            params = yaml.safe_load(f)
        
        self.R = np.array(params['R'])
        self.t = np.array(params['t'])
        
        if params.get('K_wall'):
            self.K_wall = np.array(params['K_wall'])
        if params.get('K_robot'):
            self.K_robot = np.array(params['K_robot'])
        
        print(f"参数已加载")


def main():
    print(">>> 程序启动...")
    parser = argparse.ArgumentParser(description='双相机标定工具')
    parser.add_argument('--image1', type=str, default='data/20260401-193026.jpg', help='墙装相机图像路径')
    parser.add_argument('--image2', type=str, default='data/20260401-193031.jpg', help='扫地机图像路径')
    parser.add_argument('--output', type=str, default='models/calibration.yaml', help='输出参数文件')
    parser.add_argument('--interactive', action='store_true', help='交互式验证模式')
    parser.add_argument('--calibrate', action='store_true', help='执行标定')
    parser.add_argument('--scale', type=str, default=None, help='手动输入尺度校准（像素距离,实际cm），如: --scale 1000,155.5')
    
    args = parser.parse_args()
    print(">>> 参数解析完成")
    
    print(f">>> image1={args.image1}")
    print(f">>> calibrate={args.calibrate}")
    print(f">>> interactive={args.interactive}")
    
    # 创建标定器
    calibrator = CameraCalibrator()
    
    # 手动尺度校准
    if args.scale:
        parts = str(args.scale).split(',')
        if len(parts) == 2:
            pixel_dist = float(parts[0])
            real_cm = float(parts[1])
            calibrator.set_scale(pixel_dist, real_cm)
        else:
            print("错误：尺度参数格式应为 '像素距离,实际cm'，如: --scale 1000,155.5")
    
    # 检查是否有已保存的参数
    if not args.calibrate and os.path.exists(args.output):
        print("\n=== 加载已保存的标定参数 ===")
        calibrator.load_params(args.output)
    else:
        # 执行标定
        print("\n=== 开始标定 ===")
        try:
            R, t = calibrator.calibrate(args.image1, args.image2)
        except Exception as e:
            print(f">>> 标定失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n=== 标定结果 ===")
        print(f"旋转矩阵 R:\n{R}")
        print(f"\n平移向量 t: {t.flatten()}")
        
        # 保存参数
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        calibrator.save_params(args.output)
    
    # 交互式验证
    if args.interactive:
        # 检查是否有显示器
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            print("警告：未检测到显示环境 (DISPLAY/WAYLAND_DISPLAY)")
            print("交互模式可能无法运行")
            print("尝试使用虚拟显示器或设置 DISPLAY 环境变量")
        
        print("\n=== 交互式验证模式 ===")
        print("点击墙装相机图像任意位置，获取实际坐标")
        print("按 'q' 或 'ESC' 退出")
        
        img = cv2.imread(args.image1)
        if img is None:
            print(f"无法读取图像: {args.image1}")
            return
        
        # 缩小到1/2显示（保留原始图像）
        h, w = img.shape[:2]
        img_small = cv2.resize(img, (w // 2, h // 2))
        
        # 用于存储点击的点
        click_points = []
        
        def draw_display():
            display_img = img_small.copy()
            
            # 绘制标题
            title = "Wall Camera - 点击两个点进行校准 | 按 'q' 退出 | 按 'c' 清空"
            cv2.putText(display_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制已点击的点
            for i, (px, py) in enumerate(click_points):
                color = (0, 0, 255) if len(click_points) == 2 and i == 1 else (0, 255, 0)
                cv2.circle(display_img, (px, py), 10, color, 2)
            
            # 如果有两个点，画出连线并显示距离
            if len(click_points) == 2:
                p1, p2 = click_points
                cv2.line(display_img, p1, p2, (255, 255, 0), 2)
                pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                text = f"像素距离: {pixel_dist:.1f}"
                cv2.putText(display_img, text, (mid_x - 50, mid_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            return display_img
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(click_points) < 2:
                    click_points.append((x, y))
                    
                    if len(click_points) == 2:
                        p1, p2 = click_points
                        pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        print(f">>> 两个点的像素距离: {pixel_dist:.2f}")
                        print(f">>> 请输入实际距离 (cm)，用于校准: ", end='', flush=True)
        
        cv2.namedWindow('Wall Camera')
        cv2.setMouseCallback('Wall Camera', mouse_callback)
        
        while True:
            display_img = draw_display()
            cv2.imshow('Wall Camera', display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                click_points = []
                print("\n>>> 已清空点击点")
        
        cv2.destroyAllWindows()
        
        # 计算校准
        if len(click_points) == 2:
            p1, p2 = click_points
            pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            # 实际距离 155.5 cm
            real_cm = 155.5
            calibrator.set_scale(pixel_dist, real_cm)
        
        # 继续正常的交互验证
        print("\n=== 交互式验证模式 ===")
        print("点击图像获取实际坐标，按 'q' 退出")
        
        img = cv2.imread(args.image1)
        if img is None:
            print(f"无法读取图像: {args.image1}")
            return
        
        h, w = img.shape[:2]
        img_small = cv2.resize(img, (w // 2, h // 2))
        
        click_points = []
        
        def draw_display():
            display_img = img_small.copy()
            cv2.putText(display_img, "Wall Camera - 点击获取坐标 (按 'q' 退出)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            for px, py, pcoord in click_points:
                cv2.circle(display_img, (px, py), 8, (0, 255, 0), 2)
                if pcoord:
                    text = f"({pcoord[0]:.2f}, {pcoord[1]:.2f}, {pcoord[2]:.2f})m"
                    cv2.putText(display_img, text, (px + 15, py - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return display_img
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                orig_x = x * 2
                orig_y = y * 2
                coord = calibrator.pixel_to_world(int(orig_x), int(orig_y))
                click_points.append((x, y, coord))
                
                if coord:
                    print(f">>> 点击坐标 ({orig_x}, {orig_y}) -> 实际坐标: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) 米")
                else:
                    print(f">>> 点击坐标 ({orig_x}, {orig_y}) -> 无法计算（超出视野）")
        
        cv2.namedWindow('Wall Camera')
        cv2.setMouseCallback('Wall Camera', mouse_callback)
        
        while True:
            display_img = draw_display()
            cv2.imshow('Wall Camera', display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
    
    print("\n=== 完成 ===")
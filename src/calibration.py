# -*- coding: utf-8 -*-
"""
双相机标定主程序
功能：特征匹配标定 + 坐标转换 + 交互式验证
"""

import cv2
import numpy as np
import argparse
import os
import sys
import yaml
from pathlib import Path


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
        
        # 加载配置文件（可选）
        if config:
            self.load_config(config)
        
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
        
        # 特征提取 - 使用ORB
        orb = cv2.ORB_create(nfeatures=2000)
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        print(f"图像1特征点: {len(kp1)}")
        print(f"图像2特征点: {len(kp2)}")
        
        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
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
        
        # 计算世界坐标
        P_camera = d * np.array([x_norm, y_norm, 1])
        P_world = R_inv @ P_camera + t_inv
        
        # 返回元组，确保可格式化
        return (float(P_world[0]), float(P_world[1]), float(P_world[2]))
    
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
    parser = argparse.ArgumentParser(description='双相机标定工具')
    parser.add_argument('--image1', type=str, default='data/20251203-205714.jpg', help='墙装相机图像路径')
    parser.add_argument('--image2', type=str, default='data/20251203-210012.jpg', help='扫地机图像路径')
    parser.add_argument('--output', type=str, default='models/calibration.yaml', help='输出参数文件')
    parser.add_argument('--interactive', action='store_true', help='交互式验证模式')
    parser.add_argument('--calibrate', action='store_true', help='执行标定')
    
    args = parser.parse_args()
    
    # 创建标定器
    calibrator = CameraCalibrator()
    
    # 检查是否有已保存的参数
    if not args.calibrate and os.path.exists(args.output):
        print("\n=== 加载已保存的标定参数 ===")
        calibrator.load_params(args.output)
    else:
        # 执行标定
        print("\n=== 开始标定 ===")
        R, t = calibrator.calibrate(args.image1, args.image2)
        
        print("\n=== 标定结果 ===")
        print(f"旋转矩阵 R:\n{R}")
        print(f"\n平移向量 t: {t.flatten()}")
        
        # 保存参数
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        calibrator.save_params(args.output)
    
    # 交互式验证
    if args.interactive:
        print("\n=== 交互式验证模式 ===")
        print("点击墙装相机图像任意位置，获取实际坐标")
        print("按 'q' 或 'ESC' 退出")
        
        img = cv2.imread(args.image1)
        if img is None:
            print(f"无法读取图像: {args.image1}")
            return
        
        # 缩小到1/2显示
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2))
        display_img = img.copy()
        
        cv2.putText(display_img, "Wall Camera - Click to get coordinates (Press 'q' to quit)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 用于存储所有点击的点
        click_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 坐标还原到原始图像尺寸
                orig_x = x * 2
                orig_y = y * 2
                coord = calibrator.pixel_to_world(int(orig_x), int(orig_y))
                
                # 存储点击点和坐标
                click_points.append((x, y, coord))
                
                # 在图像上标记
                cv2.circle(display_img, (x, y), 8, (0, 255, 0), 2)
                if coord:
                    text = f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})m"
                    cv2.putText(display_img, text, (x + 15, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1.5)
                    print(f">>> 点击坐标 ({orig_x}, {orig_y}) -> 实际坐标: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) 米")
                else:
                    cv2.putText(display_img, "(超出视野)", (x + 15, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1.5)
                    print(f">>> 点击坐标 ({orig_x}, {orig_y}) -> 无法计算（超出视野）")
        
        cv2.namedWindow('Wall Camera')
        cv2.setMouseCallback('Wall Camera', mouse_callback)
        
        while True:
            cv2.imshow('Wall Camera', display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
    
    print("\n=== 完成 ===")


if __name__ == '__main__':
    main()
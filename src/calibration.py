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
    
    def __init__(self, config=None, pitch=-25.37):
        self.R = None  # 旋转矩阵
        self.t = None  # 平移向量
        self.K_wall = self.DEFAULT_K  # 墙装相机内参
        self.K_robot = self.DEFAULT_K  # 扫地机内参
        self.dist = self.DEFAULT_DIST  # 畸变系数
        self.feature_matches = []
        self.scale_factor = 1.0  # 尺度因子（米/像素）
        self.camera_height = 1.74  # 相机高度（米），通过校准反推
        self.pitch = pitch  # 相机俯角（度），负值表示俯视

        # 加载配置文件（可选）
        if config:
            self.load_config(config)
        
    def set_scale(self, u1, v1, u2, v2, real_distance):
        """
        通过已知实际距离的两个像素点，反推相机有效高度。

        单目测距原理：相机高度 h 下，像素 (u,v) 的地面投影距离为
            P = (h / y_norm) * [x_norm, 1]
        给定两个像素点和它们的实际距离，可以解出 h。

        Args:
            u1, v1: 第一个校准点的像素坐标（原始图像）
            u2, v2: 第二个校准点的像素坐标（原始图像）
            real_distance: 两个点之间的实际距离（米）
        """
        K = self.K_wall

        def ground_point(u, v, h):
            xn = (u - K[0, 2]) / K[0, 0]
            yn = (v - K[1, 2]) / K[1, 1]
            lam = h / yn
            return (lam * xn, lam)

        # 计算 h=1.0 时的地面距离，距离与 h 成正比
        g1 = ground_point(u1, v1, 1.0)
        g2 = ground_point(u2, v2, 1.0)
        dist_per_h = np.sqrt((g1[0] - g2[0])**2 + (g1[1] - g2[1])**2)

        if dist_per_h > 1e-6:
            self.camera_height = real_distance / dist_per_h
            self.scale_factor = real_distance / np.sqrt((u1 - u2)**2 + (v1 - v2)**2)
            print(f">>> 校准完成: 相机有效高度 = {self.camera_height:.3f} 米")
            print(f">>> 参考: 1 像素 ≈ {self.scale_factor:.6f} 米")

            # 验证
            g1v = ground_point(u1, v1, self.camera_height)
            g2v = ground_point(u2, v2, self.camera_height)
            verify_dist = np.sqrt((g1v[0] - g2v[0])**2 + (g1v[1] - g2v[1])**2)
            print(f">>> 验证: 计算距离 = {verify_dist:.3f} 米 (目标 {real_distance} 米)")
        else:
            print(">>> 警告: 校准点选择不当（两点可能在同一水平线上）")
        
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
    
    def pixel_to_world(self, u, v, K=None):
        """
        像素坐标转世界坐标（基于相机高度和俯角的双目测距）

        原理：斜向下相机，俯角 pitch。射线与光轴夹角为 arctan(y_norm)，
        射线与水平线总角度为 arctan(y_norm) + |pitch|。
        深度 Z = camera_height / tan(arctan(y_norm) + |pitch|)

        Args:
            u, v: 像素坐标（原始图像）
            K: 相机内参矩阵（可选，默认使用墙装相机内参）

        Returns:
            (X, Y, Z): 地面坐标（米），X为横向，Y=0地面，Z为深度（相机前方为正）
        """
        if K is None:
            K = self.K_wall

        # 像素 → 归一化相机坐标
        x_norm = (u - K[0, 2]) / K[0, 0]
        y_norm = (v - K[1, 2]) / K[1, 1]

        # y_norm <= 0 表示像素在主点上方，射线不与地面相交
        if y_norm <= 0:
            return None

        # 俯角（弧度）
        pitch_rad = np.radians(abs(self.pitch))
        
        # 射线与光轴夹角
        theta = np.arctan(y_norm)
        
        # 射线与水平线总角度
        phi = theta + pitch_rad
        
        # 深度 Z = H / tan(phi)
        Z = self.camera_height / np.tan(phi)
        
        # X方向：同样需要考虑俯角
        # 射线与光轴的横向夹角为 arctan(x_norm)
        # 实际横向偏移需要考虑垂直方向的角度影响
        X = Z * np.tan(np.arctan(x_norm) * np.cos(pitch_rad))

        Y = 0.0           # 地面

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
    parser.add_argument('--camera-height', type=float, default=None, help='相机安装高度（米），用于单目测距')
    
    args = parser.parse_args()
    print(">>> 参数解析完成")
    
    print(f">>> image1={args.image1}")
    print(f">>> calibrate={args.calibrate}")
    print(f">>> interactive={args.interactive}")
    
    # 确保 data 目录存在
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        print(f">>> 创建 data 目录: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    # 创建标定器
    calibrator = CameraCalibrator()
    
    # 手动设置相机高度
    if args.camera_height is not None:
        calibrator.camera_height = args.camera_height
        print(f">>> 相机高度设置为: {args.camera_height} 米")
    
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
        
        img = cv2.imread(args.image1)
        if img is None:
            print(f"无法读取图像: {args.image1}")
            return
        
        h, w = img.shape[:2]
        img_small = cv2.resize(img, (w // 2, h // 2))
        
        # ========== 比例尺校准流程 ==========
        print("\n" + "=" * 50)
        print("=== 比例尺校准模式 ===")
        print("=" * 50)
        print("请按照以下步骤进行比例尺校准：")
        print("1. 在图像上点击两个已知实际距离的点")
        print("2. 输入这两个点之间的实际距离（厘米）")
        print("3. 程序将自动计算比例尺")
        print("=" * 50)
        
        calibration_points = []
        
        def calibration_mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                orig_x = x * 2
                orig_y = y * 2
                calibration_points.append((orig_x, orig_y))
                print(f">>> 已记录第 {len(calibration_points)} 个点: ({orig_x}, {orig_y})")
        
        cv2.namedWindow('Scale Calibration')
        cv2.setMouseCallback('Scale Calibration', calibration_mouse_callback)
        
        print("\n>>> 正在等待校准点输入（请点击两个点）...")
        
        while len(calibration_points) < 2:
            display_img = img_small.copy()
            cv2.putText(display_img, "Scale Calibration - 点击2个点", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 绘制已点击的点
            for i, (px, py) in enumerate(calibration_points):
                cv2.circle(display_img, (px // 2, py // 2), 8, (0, 255, 255), 2)
                cv2.putText(display_img, f"Point {i+1}", (px // 2 + 15, py // 2 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow('Scale Calibration', display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print(">>> 校准取消")
                cv2.destroyAllWindows()
                return
        
        cv2.destroyAllWindows()
        
        # 计算像素距离
        px1, py1 = calibration_points[0]
        px2, py2 = calibration_points[1]
        pixel_distance = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
        print(f">>> 两个校准点之间的像素距离: {pixel_distance:.2f} 像素")
        
        # 提示用户输入实际距离
        print("\n>>> 请输入这两个点之间的实际距离（米）:")
        print(">>> （可以在终端输入，或直接回车使用默认值）")
        try:
            real_distance_input = input(">>> 请输入实际距离 (米): ").strip()
            if real_distance_input:
                real_distance = float(real_distance_input)
            else:
                # 默认值：假设点击的是图像上相距约 1.55 米的点（示例）
                real_distance = 1.55
                print(f">>> 使用默认值: {real_distance} 米")
        except ValueError:
            print(">>> 输入无效，使用默认值 1.55 米")
            real_distance = 1.55
        
        # 设置比例尺（通过校准点反推相机高度）
        calibrator.set_scale(px1, py1, px2, py2, real_distance)
        print(f">>> 比例尺校准完成！")
        print("=" * 50)
        
        # ========== 正常坐标获取模式 ==========
        print("\n=== 交互式验证模式 ===")
        print("点击图像获取实际坐标，按 'q' 退出")
        
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


if __name__ == '__main__':
    main()
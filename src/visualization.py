# -*- coding: utf-8 -*-
"""
可视化模块
"""

import cv2
import numpy as np
import os


def draw_matches(img1, img2, kp1, kp2, matches, output_path=None, max_matches=50, scale=0.5):
    """
    绘制特征匹配结果
    
    Args:
        img1: 第一张图像（墙装相机）
        img2: 第二张图像（扫地机）
        kp1, kp2: 特征点
        matches: 匹配结果
        output_path: 输出路径（可选）
        max_matches: 最大显示匹配数（默认50）
        scale: 缩放比例（默认0.5，缩小到1/2显示）
    """
    # 缩放图像（用于显示）
    if scale != 1.0:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
        img2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
        # 缩放特征点坐标
        kp1_scaled = [cv2.KeyPoint(p.pt[0] * scale, p.pt[1] * scale, p.size * scale) for p in kp1]
        kp2_scaled = [cv2.KeyPoint(p.pt[0] * scale, p.pt[1] * scale, p.size * scale) for p in kp2]
    else:
        kp1_scaled = kp1
        kp2_scaled = kp2
    
    # 拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    result = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = img1
    result[:h2, w1:w1+w2] = img2
    
    # 绘制连线
    for match in matches[:max_matches]:
        x1, y1 = kp1_scaled[match.queryIdx].pt
        x2, y2 = kp2_scaled[match.trainIdx].pt
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + w1, int(y2))
        
        # 绿色连线
        cv2.line(result, pt1, pt2, (0, 255, 0), 1)
        # 红色特征点
        cv2.circle(result, pt1, 3, (0, 0, 255), -1)
        cv2.circle(result, pt2, 3, (0, 0, 255), -1)
    
    # 添加标签
    cv2.putText(result, "Wall Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(result, "Robot Camera", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 添加匹配数量
    cv2.putText(result, f"Matches: {len(matches[:max_matches])}/{len(matches)}", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"匹配结果已保存到: {output_path}")
    
    return result


def visualize_poses(img, R, t, scale=0.1):
    """
    可视化相机位姿
    
    Args:
        img: 图像
        R, t: 位姿
        scale: 箭头缩放比例
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    # 计算xyz轴方向
    axes = [
        (1, 0, 0),  # x - 红色
        (0, 1, 0),  # y - 绿色
        (0, 0, 1),  # z - 蓝色
    ]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    for axis, color in zip(axes, colors):
        direction = R @ np.array(axis) * scale
        pt2 = (int(cx + direction[0]), int(cy - direction[1]))
        cv2.arrowedLine(img, (cx, cy), pt2, color, 2)
    
    return img


def create_result_image(wall_img, robot_img, points_world):
    """
    创建结果对比图
    
    Args:
        wall_img: 墙装相机图像
        robot_img: 扫地机图像
        points_world: 世界坐标列表
    """
    h, w = wall_img.shape[:2]
    result = np.zeros((h, w * 2, 3), dtype=np.uint8)
    result[:, :w] = wall_img
    result[:, w:] = robot_img
    
    # 添加标注
    cv2.putText(result, "Wall Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Robot Camera", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result
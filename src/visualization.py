# -*- coding: utf-8 -*-
"""
可视化模块
"""

import cv2
import numpy as np


def draw_matches(img1, img2, kp1, kp2, matches):
    """
    绘制特征匹配结果
    
    Args:
        img1, img2: 两张图像
        kp1, kp2: 特征点
        matches: 匹配结果
    """
    # 拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = img1
    result[:h2, w1:w1+w2] = img2
    
    # 绘制连线
    for match in matches[:50]:  # 限制绘制数量
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + w1, int(y2))
        
        cv2.line(result, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(result, pt1, 3, (0, 0, 255), -1)
        cv2.circle(result, pt2, 3, (0, 0, 255), -1)
    
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
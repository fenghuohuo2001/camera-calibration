# -*- coding: utf-8 -*-
"""
坐标转换模块
"""

import numpy as np


def pixel_to_world_simple(u, v, fx, fy, cx, cy, camera_height, pitch=0, roll=0):
    """
    简化版坐标转换 - 已知相机高度
    
    Args:
        u, v: 像素坐标
        fx, fy: 焦距
        cx, cy: 主点
        camera_height: 相机距离地面高度（米）
        pitch, roll: 相机俯仰角和翻滚角（弧度）
        
    Returns:
        (x, y, z): 世界坐标（米）
    """
    # 角度补偿
    x_angle = np.arctan2(u - cx, fx) - pitch
    y_angle = np.arctan2(v - cy, fy) - roll
    
    if np.cos(y_angle) <= 0:
        return None  # 超出视野
    
    # 计算距离
    distance = camera_height / np.cos(y_angle)
    
    x = distance * np.sin(x_angle)
    y = distance * np.tan(y_angle)
    
    return (x, y, 0)


def pixel_to_world_stereo(u_wall, u_robot, baseline, fx, fy):
    """
    双目三角测距
    
    Args:
        u_wall: 墙装相机像素x坐标
        u_robot: 扫地机相机像素x坐标
        baseline: 两相机基线距离（米）
        fx, fy: 焦距
        
    Returns:
        depth: 深度（米）
    """
    disparity = abs(u_wall - u_robot)
    
    if disparity < 1:
        return None  # 视差太小，无法计算
    
    depth = (baseline * fx) / disparity
    
    return depth


def world_to_pixel(X, Y, Z, fx, fy, cx, cy, R, t):
    """
    世界坐标转像素坐标（反向转换）
    
    Args:
        X, Y, Z: 世界坐标
        fx, fy, cx, cy: 内参
        R, t: 外参
        
    Returns:
        (u, v): 像素坐标
    """
    # 世界 → 相机
    P_world = np.array([X, Y, Z])
    P_camera = R @ (P_world - t)
    
    # 相机 → 像素
    u = fx * P_camera[0] / P_camera[2] + cx
    v = fy * P_camera[1] / P_camera[2] + cy
    
    return (u, v)


def compute_distance(p1, p2):
    """
    计算两点间欧氏距离
    
    Args:
        p1, p2: (x, y, z) 坐标元组
        
    Returns:
        distance: 距离（米）
    """
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
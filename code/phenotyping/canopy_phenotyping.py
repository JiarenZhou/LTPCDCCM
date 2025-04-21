import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull


import matplotlib.pyplot as plt
import numpy as np


# 计算极角
def polar_angle(p0, p1):
    return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])


# 判断是否是左转
def is_left_turn(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]) > 0


# Graham扫描算法
def graham_scan(points):
    # 选择基准点
    p0_index = np.argmin(points[:, 1])
    p0 = points[p0_index]

    # 根据极角排序
    sorted_points = sorted(points, key=lambda p: polar_angle(p0, p))

    # 扫描
    stack = [sorted_points[0], sorted_points[1]]
    for i in range(2, len(sorted_points)):
        while len(stack) > 1 and not is_left_turn(stack[-2], stack[-1], sorted_points[i]):
            stack.pop()
        stack.append(sorted_points[i])

    return np.array(stack)


def calculate_crop_height(pcd):

    # 获取点云数据的坐标
    points = np.asarray(pcd.points)

    # 提取 z 轴坐标
    z_coords = points[:, 2]

    # 计算 z 轴坐标的最大值和最小值
    z_max = np.max(z_coords)
    z_min = np.min(z_coords)

    # 计算株高
    crop_height = z_max - z_min

    return crop_height

if __name__ == "__main__":

    # 读取点云
    # pcd = o3d.io.read_point_cloud("point_cloud.ply")
    pcd=o3d.io.read_point_cloud(r'H:\rice_sample.ply')
    pcd_points=np.asarray(pcd.points)

    crop_height=calculate_crop_height(pcd)
    print("株高：",abs(crop_height)/0.03)


    # 按照z轴投影到二维平面
    projected_points = pcd_points[:, :2]
    # 使用Graham扫描法找到二维凸包点
    hull = ConvexHull(projected_points)
    convex_hull_points = projected_points[hull.vertices]

    # 计算凸包点中距离最远的一对点
    max_dist = 0
    max_dist_points = None

    for i in range(len(convex_hull_points)):
        for j in range(i + 1, len(convex_hull_points)):
            dist = np.linalg.norm(convex_hull_points[i] - convex_hull_points[j])
            if dist > max_dist:
                max_dist = dist
                max_dist_points = (convex_hull_points[i], convex_hull_points[j])


    # 输出冠幅的测量值
    print("冠幅的测量值（最远点之间的欧氏距离）:", max_dist)
    # 冠幅画图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 6))
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='g', label='点云')
    plt.scatter(convex_hull_points[:, 0], convex_hull_points[:, 1], c='black', label='凸包')
    plt.plot(projected_points[hull.vertices[0], 0], projected_points[hull.vertices[0], 1], 'go')  # 绘制凸包闭合线段
    # 绘制凸包最远点之间的直线连接
    plt.plot([max_dist_points[0][0], max_dist_points[1][0]], [max_dist_points[0][1], max_dist_points[1][1]], c='orangered', label='冠幅',linestyle='--')
    for simplex in hull.simplices:
         plt.plot(projected_points[simplex, 0], projected_points[simplex, 1], 'b-')
         plt.plot(projected_points[simplex, 0], projected_points[simplex, 1], 'b-', label='凸包')
         plt.xlabel('X')
         plt.ylabel('Y')
         plt.title('Point Cloud and Convex Hull')
         plt.legend()
         plt.grid(True)
         plt.axis('equal')  # 设置XY轴等比例显示
         plt.gca().set_axis_off()  # 关闭坐标轴
         plt.axis('equal')  # 设置XY轴等比例显示
         plt.show()
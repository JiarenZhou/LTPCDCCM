import open3d as o3d
import numpy as np
import torch
import heapq
from scipy.spatial import KDTree
import networkx as nx
from scipy.spatial import ConvexHull


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def edge_estimate(points, max_nn=30, normals_raduis=10, search_raduis=10, nb_points=30, angle=90, vis=False):
    """
    :param points: 传入的点云点坐标，np格式
    :param max_nn: 邻域内最大值
    :param normals_raduis:用于HybridSearch的邻域内搜索半径
    :param search_raduis:边界提取搜索半径
    :param nb_points:边界提取领域点
    :param angle:角度阈值
    :param vis:是否可视化
    :return:
    """

    pcd = o3d.t.geometry.PointCloud(points)
    pcd.estimate_normals(max_nn=max_nn, radius=normals_raduis)  # 计算点云法向量
    boundarys, mask = pcd.compute_boundary_points(radius=search_raduis, max_nn=nb_points,
                                                  angle_threshold=angle)  # 边界提取的搜索半径、邻域最大点数和夹角阈值（角度制）

    print(f"Detect {boundarys.point.positions.shape[0]} bnoundary points from {pcd.point.positions.shape[0]} points.")
    boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
    boundarys_points = np.array(boundarys.to_legacy().points)

    if vis == True:
        pcd = pcd.paint_uniform_color([0.6, 0.6, 0.6])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='三维点云边界提取', width=1200, height=800)
        # 可视化参数设置
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # 设置背景色
        opt.point_size = 3  # 设置点的大小
        vis.add_geometry(boundarys.to_legacy())  # 加载边界点云到可视化窗口
        # vis.add_geometry(pcd.to_legacy())  # 加载原始点云到可视化窗口
        vis.run()  # 激活显示窗口，这个函数将阻塞当前线程，直到窗口关闭。
        vis.destroy_window()  # 销毁窗口，这个函数必须从主线程调用。

    return boundarys_points


def find_leaf_length(pcd,start_index,end_index):
    point_cloud_points = np.asarray(pcd.points)

    # 构建无向图
    graph = nx.Graph()

    # 添加点云中的所有点作为图的节点
    for i, point in enumerate(pcd.points):
        graph.add_node(i, position=point)

    # 构建点云中的边
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    k = 30  # 设置每个点的最近邻数
    for i, point in enumerate(pcd.points):
        _, indices, _ = kdtree.search_knn_vector_3d(point, k + 1)  # 搜索最近的k+1个点，排除自身
        for j in indices[1:]:  # 排除自身
            dist = np.linalg.norm(point - pcd.points[j])  # 计算点之间的欧几里得距离
            graph.add_edge(i, j, weight=dist)  # 添加边


    # 使用Dijkstra算法计算最短路径
    shortest_path = nx.shortest_path(graph, source=start_index, target=end_index, weight='weight')
    shortest_path_length = nx.shortest_path_length(graph, source=start_index, target=end_index, weight='weight')
    # 输出最短路径点的索引
    # print("最短路径点的索引:", shortest_path)
    # print(len(shortest_path))
    # print(shortest_path_length)

    length = pcd.select_by_index(shortest_path)
    # o3d.io.write_point_cloud('/home/zy/PycharmProjects/pythonProject/path.ply',path)
    # 获取点云的坐标数组
    points = np.asarray(length.points)
    # 生成随机颜色数组
    colors = np.random.uniform(0, 1, size=(len(points), 3))
    length.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([length])
    # o3d.visualization.draw_geometries([pcd, length])

    # # 计算最短路径的欧氏距离
    euclidean_distance = 0.0
    for i in range(len(shortest_path) - 1):
        node1 = shortest_path[i]
        node2 = shortest_path[i + 1]
        coord1 = np.array(point_cloud_points[node1])
        coord2 = np.array(point_cloud_points[node2])
        euclidean_distance += np.linalg.norm(coord2 - coord1)

    # print("The Euclidean distance of the shortest path is:", euclidean_distance)

    return euclidean_distance,shortest_path,length



def cal_leaf_width(pcd,shortest_path,leaf_length,name='',number=0):
    pcdpointcloud = np.asarray(pcd.points)
    skpointcloud = np.asarray(leaf_length.points)
    # 计算二维列表的行数
    num_rows = len(skpointcloud)
    # 计算中间行的索引
    mid_index = num_rows // 2

    point1 = np.array(skpointcloud[mid_index])
    point2 = np.array(skpointcloud[mid_index+1])
    point3 = np.array(skpointcloud[mid_index-1])
    print('point1-point2',np.linalg.norm(point1-point2))
    print('point1-point3', np.linalg.norm(point1 - point3))
    print('point2-point3', np.linalg.norm(point2 - point3))
    # 计算垂直平面的法向量
    vertical_normal1 = point2 - point1
    # 构建垂直平面方程 Ax + By + Cz + D = 0
    # 垂直平面的方程参数
    vertical_A1, vertical_B1, vertical_C1 = vertical_normal1
    vertical_D1 = -(vertical_A1 * point2[0] + vertical_B1 * point2[1] + vertical_C1 * point2[2])
    vertical_D1=vertical_D1+0.015*((vertical_A1**2+vertical_B1**2+vertical_C1**2)**0.5)
    # vertical_D = -np.dot(vertical_normal, translation_vector)
    vertical_plane_equation1 = np.array([vertical_A1, vertical_B1, vertical_C1, vertical_D1])
    vertical_normal2 = point3 - point1
    # 构建垂直平面方程 Ax + By + Cz + D = 0
    # 垂直平面的方程参数
    vertical_A2, vertical_B2, vertical_C2 = vertical_normal2
    vertical_D2 = -(vertical_A2 * point3[0] + vertical_B2 * point3[1] + vertical_C2 * point3[2])
    vertical_D2 = vertical_D2 + 0.015 * ((vertical_A2 ** 2 + vertical_B2 ** 2 + vertical_C2 ** 2) ** 0.5)
    # vertical_D = -np.dot(vertical_normal, translation_vector)
    vertical_plane_equation2 = np.array([vertical_A2, vertical_B2, vertical_C2, vertical_D2])
    print('平面1:',vertical_A1,vertical_B1,vertical_C1,vertical_D1)
    print('平面2:', vertical_A2, vertical_B2, vertical_C2, vertical_D2)
    select = []
    for i in range(pcdpointcloud.shape[0]):
        x, y, z = pcdpointcloud[i][0], pcdpointcloud[i][1], pcdpointcloud[i][2]
        distance1 = vertical_A1 * x + vertical_B1 * y + vertical_C1 * z + vertical_D1
        distance2 = vertical_A2 * x + vertical_B2 * y + vertical_C2 * z + vertical_D2
        form1 = vertical_A1 * point1[0] + vertical_B1 * point1[1] + vertical_C1 * point1[2] + vertical_D1
        form2 = vertical_A2 * point1[0] + vertical_B2 * point1[1] + vertical_C2 * point1[2] + vertical_D2
        if distance1 * form1 >= 0 and distance2 * form2 >= 0:
            select.append(i)

    ground = pcd.select_by_index(select)
    rest = pcd.select_by_index(select, invert=True)
    o3d.visualization.draw_geometries([ground])
    # o3d.io.write_point_cloud("W64A-1-1-7.21-leaf0-width.ply", ground)

    # length_save_path = '/home/zy/PycharmProjects/pythonProject/new_try/' + name + '/result/' + name + '-leafwidth_' + str(number) + '.ply'

    # o3d.io.write_point_cloud(length_save_path, ground)


    pcdpointcloud = np.asarray(ground.points)
    # 计算凸包
    hull = ConvexHull(pcdpointcloud)
    convex_hull_index=hull.vertices
    convex_hull_points=pcdpointcloud[hull.vertices]
    # print(convex_hull_index)
    # print(convex_hull_points)
    max = 0
    start = -1
    end = -1
    shortpath = []
    print('总点数:', len(convex_hull_index))
    for i in range(len(convex_hull_index)):
        for j in range(i + 1, len(convex_hull_index)):
            euclidean_distance,shortest_path,leaf_width = find_leaf_length(ground, convex_hull_index[i], convex_hull_index[j])
            # print(i, j)
            if euclidean_distance > max:
                shortpath = shortest_path
                max = euclidean_distance
                start = i
                end = j
        # 输出最短路径点的索引
    print("最短路径点的索引:", shortpath)
    print(len(shortpath))
    path = pcd.select_by_index(shortest_path)
    # 获取点云的坐标数组
    points = np.asarray(path.points)
    # 生成随机颜色数组
    colors = np.random.uniform(0, 1, size=(len(points), 3))
    path.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([path])
    o3d.visualization.draw_geometries([pcd, path])
    # euclidean_distance=find_leaf_length(path,y_min_idx,y_max_idx)
    print("叶宽是:", max)
    print("转换后叶宽是:", max / 0.03)
    print("叶宽是:", start, end)
    # print('第' + str(number) + "片叶宽是:", max / 0.03)
    return


def cal_leaf_angel(pcd,shortest_path,leaf_length):
    pcdpointcloud = np.asarray(pcd.points)
    skpointcloud = np.asarray(leaf_length.points)
    # 计算二维列表的行数
    num_rows = len(skpointcloud)

    point1 = np.array(skpointcloud[0])
    point2 = np.array(skpointcloud[int(num_rows/4)])
    point3 = np.array(skpointcloud[int(3*num_rows/4)])
    point4 = np.array(skpointcloud[int(num_rows-1)])
    vector1 = point2 - point1
    vector2 = point4 - point3

    # z轴单位向量
    z_axis = np.array([0, 0, 1])

    # 计算向量与z轴的夹角
    def calculate_angle_with_z_axis(vector):
        dot_product = np.dot(vector, z_axis)
        vector_length = np.linalg.norm(vector)
        z_axis_length = np.linalg.norm(z_axis)
        cos_theta = dot_product / (vector_length * z_axis_length)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 确保 cos_theta 在有效范围内
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    angle1 = calculate_angle_with_z_axis(vector1)
    angle2 = calculate_angle_with_z_axis(vector2)

    # print(f"point1与point2形成的向量和z轴夹角: {angle1:.2f}度")
    # print(f"point3与point4形成的向量和z轴夹角: {angle2:.2f}度")
    angle=min(angle1,angle2)
    print(f"叶片与z轴夹角: {angle:.5f}度")



if __name__ == "__main__":
    # pcd=o3d.io.read_point_cloud(r'E:\W64A-1-1-7.24/W64A-1-1-7.23-finalleaf_0.ply')
    pcd = o3d.io.read_point_cloud(r"H:\leaf_sample.ply")
    # o3d.visualization.draw_geometries([pcd],
    #                                   window_name="原点云",
    #                                   width=1024, height=768)
    points = np.asarray(pcd.points)  # 获取点坐标
    boundarys_points=edge_estimate(points)


    # 计算所有点之间的距离矩阵
    distances = np.linalg.norm(boundarys_points[:, None] - boundarys_points, axis=-1)

    # 找到距离矩阵中最大值和最小值所对应的点对
    max_distance_index = np.unravel_index(np.argmax(distances), distances.shape)
    min_distance_index = np.unravel_index(np.argmin(distances), distances.shape)

    # 将最大距离点对可视化为黑色
    max_distance_points = [boundarys_points[max_distance_index[0]], boundarys_points[max_distance_index[1]]]
    max_distance_point_cloud = o3d.geometry.PointCloud()
    max_distance_point_cloud.points = o3d.utility.Vector3dVector(max_distance_points)
    print(max_distance_point_cloud.points)
    max_distance_point_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([max_distance_point_cloud, pcd])
    # 将最小距离点对可视化为蓝色
    min_distance_points = [boundarys_points[min_distance_index[0]], boundarys_points[min_distance_index[1]]]
    min_distance_point_cloud = o3d.geometry.PointCloud()
    min_distance_point_cloud.points = o3d.utility.Vector3dVector(min_distance_points)
    print(min_distance_point_cloud.points )
    min_distance_point_cloud.paint_uniform_color([0, 0, 1])

    # 可视化点云
    o3d.visualization.draw_geometries([pcd, min_distance_point_cloud])

    # 指定起点和终点
    start_idx =  np.where((boundarys_points[max_distance_index[0]][0] == points[:, 0]) & (points[:, 1] == boundarys_points[max_distance_index[0]][1] ) & (points[:, 2] == boundarys_points[max_distance_index[0]][2] ))[0]

    end_idx = np.where((boundarys_points[max_distance_index[1]][0] == points[:, 0]) & (points[:, 1] == boundarys_points[max_distance_index[1]][1] ) & (points[:, 2] == boundarys_points[max_distance_index[1]][2] ))[0]
    print(start_idx,end_idx)


    euclidean_distance,shortest_path,leaf_length = find_leaf_length(pcd, int(start_idx), int(end_idx))
    # euclidean_distance = find_leaf_length(path, 45, 279)

    print("叶长是:", euclidean_distance / 0.03)
    # print('leaf_length',leaf_length)
    # 叶宽计算
    cal_leaf_width(pcd, shortest_path,leaf_length, name='', number=0)
    # 叶夹角计算
    cal_leaf_angel(pcd, shortest_path, leaf_length)


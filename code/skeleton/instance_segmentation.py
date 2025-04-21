import open3d as o3d
import numpy as np
import networkx as nx
from pc_skeletor import Dataset
from pc_skeletor import LBC

# ___________________________找所有点分割_____________________________
def in_range(n, start, end=0):
    return start <= n <= end if end >= start else end <= n <= start

def calculate_average_distances(points):
    distances = []
    for i in range(1, len(points) - 1):
        prev_dist = np.linalg.norm(points[i] - points[i - 1])
        next_dist = np.linalg.norm(points[i] - points[i + 1])
        avg_dist = (prev_dist + next_dist) / 2
        distances.append(avg_dist)
    return distances


def find_all_points(pcd, pcd_maize, cut_list, dist_list,percentage):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_maize)
    points = np.asarray(pcd_maize.points)
    tune = []
    num = 0
    for i in range(int(len(cut_list)*percentage)):
        dist = dist_list[cut_list[i]]
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[cut_list[i]], dist)
        num += len(idx)
        for j in idx:
            tune.append(j)
    return tune
#___________________________下采样提取骨架_____________________________
def find_sk(pcd,downsample,savepath):

    lbc = LBC(point_cloud=pcd,
              down_sample=downsample)
    lbc.extract_skeleton()#02
    lbc.extract_topology()#03
    lbc.visualize()
    lbc.show_graph(lbc.skeleton_graph)#02
    lbc.show_graph(lbc.topology_graph)#03
    lbc.save(savepath)
    return lbc
#___________________________找骨架主茎点和叶尖点_____________________________
def find_sk_point(path,lbc):

    sk02 = o3d.io.read_point_cloud(path)
    zmax = 0
    sk02 = np.asarray(sk02.points)
    zmax_lit = 0
    for i in range(sk02.shape[0]):
        if abs(sk02[i][2]) > zmax:
            zmax = abs(sk02[i][2])
            zmax_lit = i
    print('zmax_lit')
    print(zmax_lit)
    a = list(lbc.skeleton_graph.edges)  # 02
    # a=list(lbc.topology_graph.edges)#03
    b = []
    for i in range(len(a)):
        for num in a[i]:
            b.append(num)
    dict = {}
    for key in b:
        dict[key] = dict.get(key, 0) + 1

    steam_list = []
    leaf_list = []
    for key, value in dict.items():
        if value == 3:
            # 骨架主茎点
            steam_list.append(int(key))
        elif value == 1:
            # 骨架叶尖点
            leaf_list.append(int(key))
    leaf_list.remove(zmax_lit)
    steam_list.append(zmax_lit)
    print('steam_list')
    print(steam_list)
    print('leaf_list')
    print(leaf_list)
    print('all_conected')
    print(b)
    return steam_list,leaf_list,b

#___________________________找茎叶最近点 _____________________________
def findnearest(leaf_list,steam_list,b):
    G = nx.Graph()
    for i in range(0, len(b), 2):
        G.add_edge(b[i], b[i + 1], weight=1)

    leaf_all = [[0] for x in range(len(leaf_list))]
    for i in range(len(leaf_list)):
        templen = 10000
        for j in range(len(steam_list)):
            road = nx.shortest_path(G, leaf_list[i], steam_list[j], weight='weight')
            if templen > len(road):
                templen = len(road)
                leaf_all[i] = road

    steam_all = [[]]
    for i in range(len(steam_list)):
        templen = 0
        for j in range(len(steam_list)):
            road = nx.shortest_path(G, steam_list[i], steam_list[j], weight='weight')
            if templen < len(road) and i != j:
                templen = len(road)
                steam_all[0] = road
    return leaf_all,steam_all

#___________________________找所有点  _____________________________
def find_all_skpoint(pcd, rest_save_path, sk02path, steam_list, leaf_list, savepath):
    points_maize = np.asarray(pcd.points)
    o3d.io.write_point_cloud(rest_save_path, pcd)
    pcd2 = o3d.io.read_point_cloud(sk02path)
    points2 = np.asarray(pcd2.points)

    # Calculate dynamic distances
    steam_distances = calculate_average_distances(points2)
    leaf_distances = calculate_average_distances(points2)

    steam_num = 0
    for i in steam_list:
        pcd_rest_maize = o3d.io.read_point_cloud(rest_save_path)
        points_rest_maize = np.asarray(pcd_rest_maize.points)
        tune = find_all_points(pcd2, pcd_rest_maize, i, steam_distances, percentage=1)
        ground = pcd_rest_maize.select_by_index(tune)
        rest = pcd_rest_maize.select_by_index(tune, invert=True)
        tune_save_path = savepath + '/' + name + '-steam_' + str(steam_num) + '.ply'
        if len(tune) != 0:
            o3d.io.write_point_cloud(tune_save_path, ground)
        o3d.io.write_point_cloud(rest_save_path, rest)
        ground.paint_uniform_color([0, 1, 0])
        rest.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([ground, rest],
                                          window_name='steam_' + str(steam_num),
                                          width=1024, height=768,
                                          left=50, top=50,
                                          mesh_show_back_face=False)
        steam_num += 1

    leaf_num = 0
    for i in leaf_list:
        pcd_rest_maize = o3d.io.read_point_cloud(rest_save_path)
        points_rest_maize = np.asarray(pcd_rest_maize.points)
        tune = find_all_points(pcd2, pcd_rest_maize, i, leaf_distances, percentage=0.9)
        ground = pcd_rest_maize.select_by_index(tune)
        rest = pcd_rest_maize.select_by_index(tune, invert=True)
        tune_save_path = savepath + '/' + name + '-leaf_' + str(leaf_num) + '.ply'
        if len(tune) != 0:
            o3d.io.write_point_cloud(tune_save_path, ground)
        o3d.io.write_point_cloud(rest_save_path, rest)

        ground.paint_uniform_color([0, 1, 0])
        rest.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([ground, rest],
                                          window_name='leaf_' + str(leaf_num),
                                          width=1024, height=768,
                                          left=50, top=50,
                                          mesh_show_back_face=False)
        leaf_num += 1



#___________________________合并冗余点 _____________________________
def find_element(element,multi_list):
    flag=0
    for i in range(len(multi_list)):
        for j in range(len(multi_list[i])):
            if element == multi_list[i][j]:
                flag=1
                return i,j,flag
    return -1,-1,flag




def findrestpoint(name,rest_save_path,sk02path,leaf_all,steam_all):
    rest_steam = []
    rest_leaf = []
    for i in range(len(leaf_all)):
        rest_leaf.append([])

    # # # 加载两个点云数据
    pcd1 = o3d.io.read_point_cloud(rest_save_path)
    points1 = np.asarray(pcd1.points)
    pcd2 = o3d.io.read_point_cloud(sk02path)
    points2 = np.asarray(pcd2.points)
    # 初始化最近点索引列表
    closest_points_indices = []
    closest_points_dist = []

    # 遍历点云1中的每个点
    for i in range(len(points1)):
        point1 = points1[i]
        min_distance = float('inf')
        closest_point_index = -1

        # 遍历点云2中的每个点，计算距离并更新最近点信息
        for j in range(len(points2)):
            point2 = points2[j]
            distance = np.linalg.norm(point1 - point2)  # 计算欧氏距离
            if distance and distance < min_distance:
                min_distance = distance
                closest_point_index = j

        closest_points_indices.append(closest_point_index)

        closest_points_dist.append((min_distance))

    for i in range(len(closest_points_indices)):
        x1, y1, flag1 = find_element(element=closest_points_indices[i], multi_list=leaf_all)
        x2, y2, flag2 = find_element(element=closest_points_indices[i], multi_list=steam_all)

        if flag1 == 1:
            rest_leaf[x1].append(i)

        elif flag2 == 1:
            rest_steam.append(i)
        else:
            print('error', i)

    # print('rest_leaf', rest_leaf)
    # print('rest_steam', rest_steam)
    tune_save_path = savepath + '/' + name + '-steam_0.ply'
    oritune = o3d.io.read_point_cloud(tune_save_path)
    rest_steam_cloud = pcd1.select_by_index(rest_steam)
    allsteam = oritune + rest_steam_cloud
    # o3d.visualization.draw_geometries([oritune], window_name='oldsteam')
    # o3d.visualization.draw_geometries([allsteam], window_name='newsteam')
    # fianlsteam_save_path = '/home/zy/PycharmProjects/pythonProject/new_try/' + name + '/result/' + name + '_allsteam.ply'
    fianlsteam_save_path = r'/media/zy/Expansion/draw3.9/skori/'+ name +'/' + name +'_allsteam.ply'
    o3d.io.write_point_cloud(fianlsteam_save_path, allsteam)
    leaf_num = 0
    for i in rest_leaf:
        # leaf_ori_path = './new_try/' + name + '/' + name + '-leaf_' + str(leaf_num) + '.ply'
        leaf_ori_path = r'/media/zy/Expansion/draw3.9/skori/'+ name + '/' + name + '-leaf_' + str(leaf_num) + '.ply'
        pcd_leaf = o3d.io.read_point_cloud(leaf_ori_path)
        rest_leaf_cloud = pcd1.select_by_index(i)
        allleaf = pcd_leaf + rest_leaf_cloud
        # o3d.visualization.draw_geometries([pcd_leaf], window_name='orileaf_' + str(leaf_num))
        # o3d.visualization.draw_geometries([allleaf], window_name='newleaf_' + str(leaf_num))
        # fianlsteam_save_path = '/home/zy/PycharmProjects/pythonProject/new_try/' + name + '/result/' + name + '-finalleaf_' + str(leaf_num) + '.ply'
        fianlsteam_save_path = r'/media/zy/Expansion/draw3.9/skori/' + name + '/' + name + '-finalleaf_' + str(
            leaf_num) + '.ply'

        o3d.io.write_point_cloud(fianlsteam_save_path, allleaf)
        leaf_num += 1

def write_2d_list_to_file(data, filename):
    with open(filename, 'w') as file:
        for row in data:
            line = ','.join(str(element) for element in row)
            file.write(line + '\n')

def read_file_to_2d_list(filename):
    result = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = line.strip().split(',')
            row = [int(x) for x in row]
            result.append(row)
    return result


if __name__ == '__main__':

    # ___________________________下采样后提取骨架_____________________________


    # name = r'W64A-1-1-7.26'
    # name=r'A619-1-1-7.29'

    every_k_points=1
    # downsample = 0.008
    downsample=0.01

    name = r'M01-0325'

    # pointpath = r'/media/zy/Expansion/hand-cut/choose/' + name + '.ply'
    pointpath = r'/media/zy/Expansion/draw3.9/4096/4096maize/' + name + '.ply'
    pcd = o3d.io.read_point_cloud(pointpath)
    print("原始点云中点的个数为：", np.asarray(pcd.points).shape[0])
    o3d.visualization.draw_geometries([pcd])
    print("每every_k_points个点来降采样一个点")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points)
    # uni_down_pcd = pcd.voxel_down_sample(voxel_size=1)*
    print("下采样之后点的个数为：", np.asarray(uni_down_pcd.points).shape[0])
    # o3d.visualization.draw_geometries([uni_down_pcd], window_name="均匀下采样",)
    # savepath = './new_try/' + name
    savepath=r'/media/zy/Expansion/draw3.9/skori/'+name

    lbc = find_sk(uni_down_pcd, downsample, savepath)

#___________________________找骨架主茎点和叶尖点_____________________________
    # 56和69要对应
    sk02path = './new_try/' + name +'/02_skeleton_LBC.ply'
    sk02path=r'/media/zy/Expansion/draw3.9/skori/' + name +'/02_skeleton_LBC.ply'
    steam_list,leaf_list,b = find_sk_point(sk02path,lbc)


#___________________________找茎叶最近点  _____________________________
    leaf_all,steam_all = findnearest(leaf_list,steam_list,b)
    print('leaf_all')
    print(leaf_all)
    print(len(leaf_all))
    print('steam_all')
    print(steam_all)
    print(len(steam_all))
    # txt_save_path = '/home/zy/PycharmProjects/pythonProject/new_try/' + name + '/result/'
    txt_save_path=r'/media/zy/Expansion/draw3.9/skori/' +name+'/'
    write_2d_list_to_file(leaf_all, txt_save_path+'leaf.txt')
    write_2d_list_to_file(steam_all, txt_save_path+'steam.txt')


#___________________________找所有点_____________________________
    # rest_save_path = './new_try/' + name+'/rest.ply'
    rest_save_path=r'/media/zy/Expansion/draw3.9/skori/' +name+'/rest.ply'
    pcd2 = o3d.io.read_point_cloud(sk02path)
    find_all_skpoint(pcd, rest_save_path, sk02path, steam_list, leaf_list, savepath)


#___________________________合并冗余点 _____________________________
    findrestpoint(name,rest_save_path,sk02path,leaf_all,steam_all)
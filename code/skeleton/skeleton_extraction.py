import open3d as o3d
import numpy as np
import networkx as nx
# 论文可用
# 20230703
from pc_skeletor import LBC
import open3d as o3d
import numpy as np

def find_sk(pcd,downsample,savepath):

    lbc = LBC(point_cloud=pcd,
              down_sample=downsample)
    # lbc = LBC(point_cloud=pcd)
    lbc.extract_skeleton()#02
    lbc.extract_topology()#03
    lbc.visualize()
    lbc.show_graph(lbc.skeleton_graph)#02
    lbc.show_graph(lbc.topology_graph)#03
    lbc.save(savepath)
    return lbc

def find_sk_point(lbc):

    # sk02 = o3d.io.read_point_cloud(path)
    sk02 = lbc.contracted_point_cloud
    zmax = 0
    sk02 = np.asarray(sk02.points)
    zmax_lit = 0
    for i in range(sk02.shape[0]):
        if abs(sk02[i][2]) > zmax:
            zmax = abs(sk02[i][2])
            zmax_lit = i
    print('zmax_lit')
    print(zmax_lit)
    sk02 = lbc.contracted_point_cloud
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
        if value >= 3:
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


if __name__ == '__main__':
 # ___________________________提取骨架_____________________________
 #    name = r'M01_0325'
 #    name = r'D21020-1-1-7.31'
 #    name=r'D21020-1-1-7.31'
 #    name=r'A619-1-1-8.5'
    name =r'W64A-1-1-7.19'
    pointpath = r'/media/zy/Expansion/draw3.9/4096/W64A/' + name + '.ply'
    # name=r'T01_0314'
    # pointpath = r'/media/zy/KINGSTON/Maize01/' + name + '.ply'
    # pointpath=r'/media/zy/Expansion/hand-cut/'+ name + '.ply'
    # pointpath = r'/media/zy/CPBA_X64FRE/0313-1000.ply'
    pcd = o3d.io.read_point_cloud(pointpath)
    print("原始点云中点的个数为：", np.asarray(pcd.points).shape[0])
    o3d.visualization.draw_geometries([pcd])
    print("每every_k_points个点来降采样一个点")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=1)
    # uni_down_pcd = pcd.voxel_down_sample(voxel_size=0.08)
    # uni_down_pcd= pcd.farthest_point_down_sample(4096)
    print("下采样之后点的个数为：", np.asarray(uni_down_pcd.points).shape[0])
    o3d.visualization.draw_geometries([uni_down_pcd],
                                      window_name="均匀下采样",
                                      width=1200, height=800,
                                      left=50, top=50)
    # uni_down_pcd = pcd.voxel_down_sample(voxel_size=0.5)
    #
    # savepath = './new_try/' + name
    savepath=r'/media/zy/Expansion/draw3.9/skori/'+ name
    savepath=r'/media/zy/Expansion/draw3.9/sk4096/'+name
    downsample = 0.008
    # downsample=-1
    # find_sk(uni_down_pcd, downsample, savepath)
    lbc = find_sk(uni_down_pcd, downsample, savepath)
    print(lbc.skeleton_graph)
    print(lbc.skeleton_graph.nodes)
    print(lbc.skeleton_graph.edges)

    # steam_list, leaf_list, b = find_sk_point(lbc)


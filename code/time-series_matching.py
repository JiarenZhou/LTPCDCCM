import os
import random
import shutil
import numpy as np
import open3d as o3d
# import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

def calculate_centroid(file_path):
    # 读取txt文件
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label'])

    # 计算质心
    centroid = data[['x', 'y', 'z']].mean().to_dict()
    return centroid

def split_txt_by_label(file_path):
    # 读取txt文件
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label'])

    # 获取唯一的标签
    labels = data['label'].unique()

    # 按标签划分数据
    split_data = {label: data[data['label'] == label] for label in labels}

    return labels,split_data

def calculate_centroids(split_data):
    centroids = {}
    for label, df in split_data.items():
        # 计算质心
        centroid = df[['x', 'y', 'z']].mean().to_dict()
        centroids[label] = centroid
    return centroids


def calculate_distance_matrix(centroids1, centroids2):
    # 获取所有质心的坐标
    coords1 = np.array([list(c.values()) for c in centroids1.values()])
    coords2 = np.array([list(c.values()) for c in centroids2.values()])
    print('coords1:',np.shape(coords1))
    print('coords2:', np.shape(coords2))

    # 计算距离矩阵
    distance_matrix = cdist(coords1, coords2)
    print('distance_matrix:', np.shape(distance_matrix))
    return distance_matrix

# def find_min_distance_labels(distance_matrix, labels1, labels2):
#     min_distance_labels = []
#     for i, row in enumerate(distance_matrix):
#         sorted_indices = np.argsort(row)
#         min_index = sorted_indices[0]
#
#         # 检查最小值对应的标签
#         if (int(labels1[i]) == 0 and int(labels2[min_index]) != 0) or (int(labels1[i]) != 0 and int(labels2[min_index]) == 0):
#             # 如果最小值对应的标签有一个是0.0，取第二小的值
#             min_index = sorted_indices[1]
#
#         min_distance_labels.append((labels1[i], labels2[min_index], row[min_index]))
#     return min_distance_labels
def find_min_distance_labels(distance_matrix, labels1, labels2, centroids1, centroids2, threshold):
    min_distance_labels = []
    for i, row in enumerate(distance_matrix):
        sorted_indices = np.argsort(row)
        min_index = sorted_indices[0]

        # 检查最小值对应的标签
        if (int(labels1[i]) == 0 and int(labels2[min_index]) != 0) or (int(labels1[i]) != 0 and int(labels2[min_index]) == 0):
            # 如果最小值对应的标签有一个是0.0，取第二小的值
            min_index = sorted_indices[1]

        min_distance = row[min_index]
        min_distance_labels.append((labels1[i], labels2[min_index], min_distance))

    new_leaves = []
    old_leaves = []
    for label1, label2, distance in min_distance_labels:
        centroid1 = centroids1[label1]
        centroid2 = centroids2[label2]
        if distance > threshold and centroid2['z'] > np.mean([c['z'] for c in centroids2.values()]):
            new_leaves.append((label2, distance))
        elif distance < threshold and centroid1['z'] < np.mean([c['z'] for c in centroids1.values()]):
            old_leaves.append((label1, distance))
        else:
            print(f"Label pair: ({label1}, {label2}), Distance: {distance:.2f}")

    final_new_leaves=[]
    final_old_leaves=[]
    # 计算label数量差值
    label_diff = len(labels2) - len(labels1)

    if label_diff > 0:  # 有新生叶片
        # 选择现有的new_leaves中距离最远的n个叶片视为新生叶片
        final_new_leaves = sorted(new_leaves, key=lambda x: x[1], reverse=True)[:label_diff]

    elif label_diff < 0:  # 有脱落叶片
        # 选择现有的old_leaves中距离最远的n个叶片视为新生叶片
        final_old_leaves = sorted(old_leaves, key=lambda x: x[1], reverse=True)[:abs(label_diff)]

    return min_distance_labels, final_new_leaves, final_old_leaves

if __name__ == "__main__":
    file_path1 = r'D:\Data\matching/D21116-10-1-8.1.txt'
    file_path2 = r'D:\Data\matching/D21116-10-1-8.2.txt'

    labels1, split_data1 = split_txt_by_label(file_path1)
    labels2, split_data2 = split_txt_by_label(file_path2)
    print(sorted(labels1))
    print(sorted(labels2))
    # print(type(labels1[0]))
    centroids1 = calculate_centroids(split_data1)
    centroids2 = calculate_centroids(split_data2)

    # print("Centroids from file 1:")
    # print(centroids1)
    # print("Centroids from file 2:")
    # print(centroids2)

    distance_matrix = calculate_distance_matrix(centroids1, centroids2)
    print("Distance Matrix:")
    # print(distance_matrix)
    #玉米大豆6*0.03，水稻3*0.03 0.03是点云和实际比例 1cm=0.03
    threshold = 6*0.03
    min_distance_labels,new_leaves, old_leaves = find_min_distance_labels(distance_matrix, labels1, labels2, centroids1, centroids2, threshold)
    print("Label pairs with minimum distances:")
    for label1, label2, distance in min_distance_labels:
        print(f"Label pair: ({label1}, {label2}), Distance: {distance:.2f}")
    print("Newly sprouted leaves:")
    for label, distance in new_leaves:
        print(f"Label: {label}, Distance: {distance:.2f}")

    print("Old leaves about to fall:")
    for label, distance in old_leaves:
        print(f"Label: {label}, Distance: {distance:.2f}")

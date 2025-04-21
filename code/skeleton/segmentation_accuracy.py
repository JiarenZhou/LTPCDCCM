
import numpy as np
import open3d as o3d
import numpy as np



def calArray2dDiff(array_0, array_1):
    array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
    array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])

    return np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])

def calArrayinDiff(array_0, array_1):
    array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
    array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])

    return np.in1d(array_0_rows, array_1_rows)


if __name__ == '__main__':
    print()

    name='W64A-1-1-7.31'
    number='6'



    name1=name[-4:]+'-'+number

    # path1 = 'D:/maize-skeleton/workshop/new_try/'+ name+'/result/'+name+'-finalleaf_'+number+'.ply'
    path1 = 'D:/maize-skeleton/workshop/new_try/'+name+'/result/'+name+'_allsteam.ply'
    point_cloud1 = o3d.io.read_point_cloud(path1)
    points1 = np.asarray(point_cloud1.points)

    # path2 = r'D:\maize-skeleton\workshop\ac_cal/'+name1+'.ply'
    path2 = r'D:\maize-skeleton\workshop\ac_cal/'+name[-4:]+'-steam.ply'
    point_cloud2 = o3d.io.read_point_cloud(path2)
    points2 = np.asarray(point_cloud2.points)

    print(len(points1))
    print(len(points2))
    # 在1不在2
    print(len(calArray2dDiff(points1, points2)))
    # 在2不在1
    print(len(calArray2dDiff(points2, points1)))
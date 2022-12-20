import time

start_all = time.time()
# import configparser
import sys
import numpy as np
import matplotlib.pyplot as plt  # 绘图第三方库
# import mmap
import os
# from pandas import DataFrame as pandas_DataFrame
import configparser
import xlrd

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'----------------------------------------0.内函数----------------------------------------'
def readSectionElem(FileName, SectName):
    data_dm = xlrd.open_workbook(FileName)
    # table_dm = data_dm.sheet_by_index(0)
    # table_dm = data_dm.sheet_by_name('design')
    table_dm = data_dm.sheet_by_name(SectName)
    nrows_dm = table_dm.nrows
    Secdata = np.array([table_dm.row_values(i) for i in range(nrows_dm)])
    SectDesign = Secdata[1:, 1:6]
    SectElem = np.array([float(x) for row in SectDesign for x in row]).reshape(SectDesign.shape)
    OffsetVec = Secdata[1, 6:8]
    OffsetVec = np.array([float(OffsetVec[0]), float(OffsetVec[1])])
    return SectElem, OffsetVec;

# 0.1 数据读取(文本或las)
def read_data(DataFileName, time_proportion: '按照时间先后的百分比取点' = 1, maxRange=0, if_unique=False):
    maxRange = maxRange * 1000
    try:
        start = time.time()
        print('《《《开始读取数据》》》')
        PointXYZ = np.load(DataFileName)  # 单位mm
        lineread = time.time()
        print("%.1f万点数据npy方法读取耗时：%.2f秒" % (PointXYZ.shape[0] / 10000, (lineread - start)))
        lineread = time.time()
        # 数据筛选
        point_num = int(PointXYZ.shape[0] * time_proportion)  # 取前time_proportion时间内的数据
        PointXYZ = PointXYZ[0:point_num * 1, :]
        PointXYZ = PointXYZ[PointXYZ[:, 0] > 1000, :]  # 删除1m内的点
        PointXYZ = PointXYZ[(np.abs(PointXYZ[:, 0]) <= maxRange) &
                            (np.abs(PointXYZ[:, 1]) <= maxRange * np.tan(np.pi / 4)) &
                            (np.abs(PointXYZ[:, 2]) <= maxRange * np.tan(np.pi / 4)), :]  # 删除最大包围盒外的点
        if if_unique == True:
            [PointXYZ, PointXYZ_weighted] = np.unique(PointXYZ, axis=0, return_counts=True)  # 删除重复点
            start = time.time()
            print("去重耗时：%.2f秒" % (start - lineread))
    except:
        raise ValueError('ERROR:Read File Error')
    return PointXYZ




# 画2d投影图

def tunnelSectionDesign(SectElem, TranVec, RotaMat):
    m, n = SectElem.shape
    SectElem = SectElem * np.array([1, 1, 1, 1, np.pi/180])
    SectElem = np.vstack((SectElem, np.array([0, 0, 0, 0, np.pi/2])))

    x = []
    y = []
    for i in range(m):
        if SectElem[i, 0] == 1:
            t = np.linspace(SectElem[i, 4], SectElem[i+1, 4], num=10001)
            xt = SectElem[i, 1] + SectElem[i, 3]*np.cos(t)
            yt = SectElem[i, 2] + SectElem[i, 3]*np.sin(t)

        # 直线段
        elif SectElem[i, 0] == 2:
            xt = SectElem[i, 1]
            yt = SectElem[i, 2]

        x = np.hstack((x, xt)).tolist()
        y = np.hstack((y, yt)).tolist()

    x = np.hstack((x, -1*np.array(list(reversed(x)))))
    y = np.hstack((y, np.array(list(reversed(y)))))
    xy = np.vstack((x, y))

    #plt.plot(xy[0, :], xy[1, :], c='purple')

    return xy


def plot_2d(x_show, y_show, title='polyfitting', point_type='b.', label='original values'):
    plt.plot(x_show, y_show, point_type, label=label, linewidth=0.1)
    plt.xlabel('x axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title(title)
    plt.ion()
    # plt.savefig('%s.jpg'%title)
    plt.show()
    return


# 0.4轴线拟合，输入点云数据，桩号范围skn_range(第一个值为小桩号，第二个值为大桩号，第三四分别为fyx和fzx的导数次数1或2)
# 通过单侧规则边缘与加常数代替中轴线
# 和VFH比较，在判断仪器左右时，当5m处0点没有坐标，转而就按10m处的0点坐标
def Axis_fitting_VTsShan(PointXYZ, skn_range_yx, skn_range_zx, fyx=2, fzx=1,LR = 'L'):
    skn_range = np.array(skn_range_yx) + 0
    if skn_range_yx[0] > skn_range_zx[0]:
        skn_range[0] = skn_range_zx[0]
    if skn_range_yx[1] < skn_range_zx[1]:
        skn_range[1] = skn_range_zx[1]
    pointxyz_index = np.where((PointXYZ[:, 0] >= skn_range[0]) & (PointXYZ[:, 0] <= skn_range[1]))[0]
    PointXYZ2 = PointXYZ[pointxyz_index, :]  # 所有range范围内的点
    xyzforfit_max = []  # 所有最大值
    xyzforfit_min = []  # 所有最小值
    # 小skn-大skn米处轴线拟合
    # time1=time.time()
    for i in np.arange(skn_range[0], skn_range[1], 0.1):
        itertime = round(i, 1)
        if i == skn_range[0]:
            index = np.where((PointXYZ2[:, 0] >= itertime) & (PointXYZ2[:, 0] <= itertime + 0.1))[0]
        else:
            index = np.where((PointXYZ2[:, 0] > itertime) & (PointXYZ2[:, 0] <= itertime + 0.1))[0]
        dx = PointXYZ2[index, :]
        try: #todo 此处没有点的时候， sknrange超出实际点的范围时，报错
            xyzforfit_max.append(dx.max(0))
        except:
            print('1')
        xyzforfit_min.append(dx.min(0))
    xyzforfit = np.vstack([xyzforfit_max, xyzforfit_min])

    # time2=time.time()
    # print(time2-time1)
    xyzforfit_max = np.array(xyzforfit_max)
    xyzforfit_min = np.array(xyzforfit_min)
    # 自动判断仪器在隧道壁左侧还是右侧
    myD = 5 #距离仪器的距离，判断距离
    zD = 0.1 #距离仪器的高度
    # # todo 自动判断LR的代码需要修改
    if 'LR' not in vars():
        print('仪器所在位置未确定')
    elif LR == 'L':#左侧，计算下面
        yxforfit_index = np.where((xyzforfit_min[:, 0] >= skn_range_yx[0]) & (xyzforfit_min[:, 0] <= skn_range_yx[1]))[0]
        yxforfit = xyzforfit_min[yxforfit_index, :]
    else:#右侧，计算上面
        yxforfit_index = np.where((xyzforfit_max[:, 0] >= skn_range_yx[0]) & (xyzforfit_max[:, 0] <= skn_range_yx[1]))[0]
        yxforfit = xyzforfit_max[yxforfit_index, :]
    zxforfit_index = np.where((xyzforfit_max[:, 0] >= skn_range_zx[0]) & (xyzforfit_max[:, 0] <= skn_range_zx[1]))[0]
    zxforfit = xyzforfit_max[zxforfit_index, :]
    # 根据yx和zx的范围进行拟合
    dy = np.polyfit(yxforfit[:, 0], yxforfit[:, 1], fyx)
    dz = np.polyfit(zxforfit[:, 0], zxforfit[:, 2], fzx)
    py = np.poly1d(dy)
    deltd = py(0)/2
    if LR == 'L':
        dy[-1] = dy[-1]-deltd#+0.07/2#另外加减仪器中心距离墙壁的距离
    else:
        dy[-1] = dy[-1]-deltd#-0.07/2

    return dy, dz, xyzforfit

def Coord_Trans(XYZ_0, R_diag, TranVec, type):
    if type == 1:  # 坐标正算
        XYZ_1 = np.dot(R_diag, XYZ_0.T) + TranVec
    elif type == 2:  # 坐标反算
        XYZ_1 = np.dot(np.linalg.inv(R_diag), (XYZ_0.T - TranVec))
    XYZ_1 = XYZ_1.T
    return XYZ_1

# 0.6 超欠挖批量计算，新版
def OverBreak(PointXYZ, dy, dz, skn_range, SectElem, resolution=0.1,
              deltH=-2.6,
              delt_inner_radius=-0.05,
              StdLine=-0.6,
              is_test=False):
    '———————————————————————————————— 1 计算所有点云方位角————————————————————————————————————'
    # 计算所有点云方位角
    d = np.sqrt(PointXYZ[:, 1] ** 2 + PointXYZ[:, 2] ** 2)  # 所有点云距离
    azimuthAll = np.arccos(PointXYZ[:, 1] / d) # 所有点云方位角
    '———————————————————————————————— 2 计算断面设计交点————————————————————————————————————'
    # 计算所有右侧的断面设计交点
    SectElem = np.vstack([SectElem, SectElem[-1:, :]])
    SectElem[-1:, -1] = 90  # 增加一行90°的
    interPoint = []  # 交点
    for i in range(np.size(SectElem[:, 0])):
        # interPoint格式为交点方位角、圆心点XY、圆半径R
        if SectElem[i, 4] > 0 and SectElem[i, 4] < 90:
            X2 = np.cos(np.deg2rad(SectElem[i, 4])) * SectElem[i, 3] + SectElem[i, 1]  # X坐标
            Y2 = np.sin(np.deg2rad(SectElem[i, 4])) * SectElem[i, 3] + SectElem[i, 2]  # Y坐标
            D = np.sqrt(X2 ** 2 + Y2 ** 2)  # 交点距离
            azimuth = np.arcsin(Y2 / D)
            interPoint.append([azimuth, SectElem[i, 1], SectElem[i, 2], SectElem[i, 3]])
        elif SectElem[i, 4] == 90:
            X2 = 0  # X坐标
            Y2 = SectElem[i, 3] + SectElem[i, 2]  # Y坐标
            D = np.sqrt(X2 ** 2 + Y2 ** 2)  # 交点距离
            azimuth = np.pi / 2
            interPoint.append([azimuth, SectElem[i, 1], SectElem[i, 2], SectElem[i, 3]])

    '———————————————————————————————— 3 循环计算所有点云的超欠挖值————————————————————————————————————'
    # 由右侧的断面设计交点，所有点云的超欠挖值
    interPoint = np.array(interPoint)
    overBreakValue = PointXYZ[:, 0:1] * 0  # 超欠挖值
    for i in range(len(interPoint)):
        if i == 0:  # 第一个方位角
            start1 = 0
        else:  # 第N个方位角
            start1 = interPoint[i - 1, 0]
        end1 = interPoint[i, 0]
        start2 = np.pi - end1
        end2 = np.pi - start1
        index_temp = np.where(((azimuthAll >= start1) & (azimuthAll < end1)) | \
                              ((azimuthAll >= start2) & (azimuthAll < end2)))[0]
        overBreakValue[index_temp, :] = np.sqrt(
            (np.abs(PointXYZ[index_temp, 1:2]) - interPoint[i, 1]) ** 2 +
            (np.abs(PointXYZ[index_temp, 2:3]) - interPoint[i, 2]) ** 2) - interPoint[i, -1]  # 超欠挖值
    overBreakValue[azimuthAll == np.pi, :] = PointXYZ[azimuthAll == np.pi, 2:3] - interPoint[-1, -1]
    # overBreakValue
    return overBreakValue, azimuthAll


def modfy_config(dy,dz,ini_Filename):
    try:
        config = configparser.ConfigParser()
        # -read读取ini文件
        config.clear()
        for encoding in 'GB18030', 'UTF-8-sig':
            try:
                config.read(ini_Filename, encoding=encoding)
            except:
                continue
        # 如果检测没有问题，config对象 DY DZ 写入
        config.set('self_config', 'dy', str(dy.tolist())[1:-1])  # 给分组设置值
        config.set('self_config', 'dz', str(dz.tolist())[1:-1])  # 给分组设置值
        o = open(ini_Filename, 'w')
        config.write(o)
        o.close()
        return True
    except Exception as e:
        return False


def check_plot(data_Filename,horizontal_value,skn_range_yx,skn_range_zx,LR,ini_Filename,Para_FileName):
    '''
    :param data_Filename: '要处理的文件全路径名'
    :return:
    '''
    '----------------------------------------0.配置文件读取----------------------------------------'
    # filePath = data_Filename[:data_Filename.rfind('\\', 0, data_Filename.rfind('\\')) + 1]

    # Para_FileName = 'parameter.xlsx'  # 平竖曲线断面参数表
    # ini_Filename = 'config.ini' # 配置文件地址
    horizontalValue = horizontal_value.split(',')
    # print(horizontal_value)
    # 创建文件夹
    print('《《《开始读取配置文件》》》')
    # 实例化configParser对象
    config = configparser.ConfigParser()
    # -read读取ini文件
    config.clear()
    for encoding in 'GB18030', 'UTF-8-sig':
        try:
            config.read(ini_Filename, encoding=encoding)
        except:
            continue
    skn_range = list(map(float, config.get('self_config', 'skn_range').split(",")))
    ori_high = config.getfloat('self_config', 'ori_high')
    StdLine = config.getfloat('self_config', 'StdLine')
    fyx = config.getint('self_config', 'fyx')
    fzx = config.getint('self_config', 'fzx')
    resolution = config.getfloat('self_config', 'resolution')
    deltH = config.getfloat('self_config', 'delth')
    delt_inner_radius = config.getfloat('self_config', 'delt_inner_radius')
    dy = config.get('self_config', 'dy')
    dz = config.get('self_config', 'dz')
    old_horizontal_value = config.get('self_config', 'old_horizontal_value')
    '----------------------------------------1.数据读取----------------------------------------'
    is_test = True
    maxRange = max(max(skn_range), max(skn_range_yx), max(skn_range_zx))  # 单位m
    PointXYZ = read_data(data_Filename, maxRange=maxRange, if_unique=True)[:, :4]
    PointXYZ = PointXYZ.astype('float64')
    PointXYZ[:,:3] = PointXYZ[:,:3]/1000
    PointXYZ[:, 3] = PointXYZ[:, 3] / 50  # 反射率以由1-100转换为0-1

    '----------------------------------------2.点云的投影关系----------------------------------------'
    if '3GGD' in data_Filename: # 如果是70，则旋转点云180°
        #旋转
        gamma = np.pi
        R = np.array([[1, 0, 0],
                       [0, np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma), np.cos(gamma)]])
        T = np.zeros([3,1])
        PointXYZ[:,:3] = Coord_Trans(PointXYZ[:,:3], R, T, 1)
    gamma = -np.deg2rad(float(horizontalValue[0]))
    beta = -np.deg2rad(float(horizontalValue[1]))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(gamma), -np.sin(gamma)],
                   [0, np.sin(gamma), np.cos(gamma)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    R = np.dot(Ry, Rx)  # 先转x再y
    T = np.zeros([3, 1])
    PointXYZ[:, :3] = Coord_Trans(PointXYZ[:, :3], R, T, 1)

    step_size = int(PointXYZ.shape[0] / 100000)
    x_down = PointXYZ[::step_size, 0]
    y_down = PointXYZ[::step_size, 1]
    z_down = PointXYZ[::step_size, 2]

    dy, dz, xyzforfit = Axis_fitting_VTsShan(PointXYZ, skn_range_yx, skn_range_zx, fyx=fyx, fzx=fzx,LR=LR)  # 二衬计算

    '----------------------------------------5.拟合线与点云的投影关系----------------------------------------'
    pz = np.poly1d(dz)
    z_poly1d = pz(x_down)  # 该步骤和np.polyval(z1, x_down)效果相同
    py = np.poly1d(dy)
    y_poly1d = py(x_down)  # 该步骤和np.polyval(y1, x_down)效果相同
    plt.close()
    plt.figure('轴线拟合关系')
    # plt.subplot(2, 1, 1)
    # plot_2d(x_down, y_down, title=' y-polyfitting', point_type='b.', label='original values')
    plt.plot(x_down, y_down,'b.', label='original values',linewidth=0.1)
    plt.plot(x_down, y_poly1d,'r.', label='polyfit values',linewidth=0.1)
    if 'xyzforfit' in locals():
        plt.plot(xyzforfit[:, 0], xyzforfit[:, 1], 'g.', label='mean values')
    plt.xlabel('x axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title('点云俯视图--轴线拟合')
    plt.savefig('image/point_cloud_look_down_line.jpg')
    plt.close()

    plt.figure('轴线拟合关系')
    # plt.subplot(2, 1, 2)
    plt.plot(x_down, z_down, 'b.', label='original values')
    plt.plot(x_down, z_poly1d,'r.', label='polyfit values')
    if 'xyzforfit' in locals():
        plt.plot(xyzforfit[:, 0], xyzforfit[:, 2], 'g.', label='mean values')
    plt.xlabel('x axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title('点云侧视图--轴线拟合')
    plt.savefig('image/point_cloud_side_look_line.jpg')
    plt.close()
    print(dy,dz)
    return PointXYZ,dy,dz


def plot_section(PointXYZ,ini_Filename,Para_FileName,dy,dz):
    '----------------------------------------主方向坐标转换----------------------------------------'
    # todo 加入曲线坐标转换
    config = configparser.ConfigParser()
    # -read读取ini文件
    config.clear()
    for encoding in 'GB18030', 'UTF-8-sig':
        try:
            config.read(ini_Filename, encoding=encoding)
        except:
            continue
    skn_range = list(map(float, config.get('self_config', 'skn_range').split(",")))
    ori_high = config.getfloat('self_config', 'ori_high')
    StdLine = config.getfloat('self_config', 'StdLine')
    fyx = config.getint('self_config', 'fyx')
    fzx = config.getint('self_config', 'fzx')
    resolution = config.getfloat('self_config', 'resolution')
    deltH = config.getfloat('self_config', 'delth')
    delt_inner_radius = config.getfloat('self_config', 'delt_inner_radius')

    # 直线坐标转换
    PointXYZ_Trans = np.hstack(
        [PointXYZ[:, 0:1], PointXYZ[:, 1:2] - dy[1], PointXYZ[:, 2:3] - dz[1] + ori_high, PointXYZ[:, 3:4]])
    angle_dy = np.arctan(dy[0])
    angle_dz = np.arctan(dz[0])
    Rz = np.array([[np.cos(angle_dy), np.sin(angle_dy), 0],
                   [-np.sin(angle_dy), np.cos(angle_dy), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(angle_dz), 0, np.sin(angle_dz)],
                   [0, 1, 0],
                   [-np.sin(angle_dz), 0, np.cos(angle_dz)]])
    R = np.dot(Ry, Rz)  # 先转z再y
    T = np.zeros([3, 1])
    PointXYZ_Trans[:, :3] = Coord_Trans(PointXYZ_Trans[:, :3], R, T, 1)
    PointXYZ_Trans = PointXYZ_Trans[PointXYZ_Trans[:, 2] >= 0, :]
    '----------------------------------------6.超欠挖计算----------------------------------------'
    SectElem, OffsetVec = readSectionElem(Para_FileName, 'design')
    print('《《《开始超欠挖计算》》》')
    overBreakValue, azimuthAll = OverBreak(PointXYZ_Trans, dy, dz, skn_range, SectElem, resolution=resolution,
                                           deltH=deltH, delt_inner_radius=delt_inner_radius,
                                           is_test=False, StdLine=StdLine)  # 断面平移高度-2.5m
    # jgAll数据顺序（x,y,z, 方向角 ,超欠挖值  ,f）
    azimuthAll = azimuthAll.reshape(-1, 1)
    jgAll = np.hstack([PointXYZ_Trans[:, :3], azimuthAll, overBreakValue, PointXYZ_Trans[:, -1:]])
    jgAll = jgAll[jgAll[:, -2] > delt_inner_radius, :]  # 去除内轮廓内点云
    # jg数据顺序（x,y,z,θ,超欠挖值,f）

    '----------------------------------------断面切片检查----------------------------------------'

    avg_len = (np.max(jgAll[:, 0]) - np.min(jgAll[:, 0]))/6
    ii = 1
    for i in np.arange(np.min(jgAll[:, 0]), np.max(jgAll[:, 0]), avg_len):
        print(i)
        f2 = plt.figure('第' + str(round(i, 3)) + 'm断面切片')
        xy = tunnelSectionDesign(SectElem, 0, 0)
        plt.plot(xy[0, :], xy[1, :], c='g', label='设计断面')
        # plt.plot(jgAll[:, 0], jgAll[:, 1], c='b')
        point_plt = jgAll[(jgAll[:, 0] > (i - resolution / 2)) & (jgAll[:, 0] < (i + resolution / 2)), :]
        plt.scatter(point_plt[:, 1], point_plt[:, 2], s=0.8, c=point_plt[:, 4], marker='.', label='测量断面点云')
        plt.axis('equal')
        plt.savefig('image/section_%d.jpg'%ii)
        plt.close()
        ii+=1
    '----------------------------------------三维展示检查----------------------------------------'

    f1 = plt.figure('三维展示图', figsize=(10, 6))
    if jgAll.shape[0] == 0:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.text(0.5, 0.5, s='点云数为0或其他错误', ha='center', va='center')
    else:
        jgAll_shape = round(jgAll.shape[0] / 5000)
        ax = plt.axes(projection='3d')
        ax.scatter3D(jgAll[::jgAll_shape, 0], jgAll[::jgAll_shape, 1], jgAll[::jgAll_shape, 2],
                     c=jgAll[::jgAll_shape, 4], marker='.')
        max_len = np.nanmax(jgAll[:, 0]) - np.nanmin(jgAll[:, 0])
        ax.set(xlabel='X',
               ylabel='Y',
               zlabel='Z',
               xticks=np.arange(np.nanmin(jgAll[:, 0]), np.nanmax(jgAll[:, 0]), 5),
               yticks=np.arange(np.nanmin(jgAll[:, 1]) - max_len / 2, np.nanmax(jgAll[:, 1]) + max_len / 2, 5),
               zticks=np.arange(np.nanmin(jgAll[:, 2]) - max_len / 2, np.nanmax(jgAll[:, 2]) + max_len / 2, 5)
               )  # 设置坐标轴格式
    plt.savefig('image/3d_show.jpg')
    print('测试总共用时%.2f秒' % (time.time() - start_all))


def plot_all(data_Filename,horizontal_value):
    '''
    :param data_Filename: '要处理的文件全路径名'
    :return:
    '''
    '----------------------------------------0.配置文件读取----------------------------------------'
    filePath = os.path.dirname(os.path.dirname(data_Filename))
    # print(filePath)
    Para_FileName = filePath+'\\parameter.xlsx' # 平竖曲线断面参数表
    ini_Filename = filePath+'\\config.ini'  # 配置文件地址
    horizontalValue = horizontal_value.split(',')
    # print(Para_FileName,ini_Filename)
    # 创建文件夹
    print('《《《开始读取配置文件》》》')
    # 实例化configParser对象
    config = configparser.ConfigParser()
    # -read读取ini文件
    config.clear()
    for encoding in 'GB18030', 'UTF-8-sig':
        try:
            config.read(ini_Filename, encoding=encoding)
        except:
            continue
    skn_range = list(map(float, config.get('self_config', 'skn_range').split(",")))
    skn_range_yx = list(map(float, config.get('self_config', 'skn_range_yx').split(",")))
    skn_range_zx = list(map(float, config.get('self_config', 'skn_range_zx').split(",")))
    ori_high = config.getfloat('self_config', 'ori_high')
    StdLine = config.getfloat('self_config', 'StdLine')
    fyx = config.getint('self_config', 'fyx')
    fzx = config.getint('self_config', 'fzx')
    resolution = config.getfloat('self_config', 'resolution')
    deltH = config.getfloat('self_config', 'delth')
    delt_inner_radius = config.getfloat('self_config', 'delt_inner_radius')
    dy = config.get('self_config', 'dy')
    dz = config.get('self_config', 'dz')
    LR = config.get('self_config', 'LR')

    old_horizontal_value = config.get('self_config', 'old_horizontal_value')
    '----------------------------------------1.数据读取----------------------------------------'
    is_test = True
    maxRange = max(max(skn_range), max(skn_range_yx), max(skn_range_zx))  # 单位m
    PointXYZ = read_data(data_Filename, maxRange=maxRange, if_unique=True)[:, :4]
    PointXYZ = PointXYZ.astype('float64')
    PointXYZ[:,:3] = PointXYZ[:,:3]/1000
    PointXYZ[:, 3] = PointXYZ[:, 3] / 50  # 反射率以由1-100转换为0-1

    '----------------------------------------2.点云的投影关系----------------------------------------'
    if '3GGD' in data_Filename: # 如果是70，则旋转点云180°
        #旋转
        gamma = np.pi
        R = np.array([[1, 0, 0],
                       [0, np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma), np.cos(gamma)]])
        T = np.zeros([3,1])
        PointXYZ[:,:3] = Coord_Trans(PointXYZ[:,:3], R, T, 1)
    gamma = -np.deg2rad(float(horizontalValue[0]))
    beta = -np.deg2rad(float(horizontalValue[1]))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(gamma), -np.sin(gamma)],
                   [0, np.sin(gamma), np.cos(gamma)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    R = np.dot(Ry, Rx)  # 先转x再y
    T = np.zeros([3, 1])
    PointXYZ[:, :3] = Coord_Trans(PointXYZ[:, :3], R, T, 1)

    #三维显示
    import open3d as o3d
    PCD = o3d.geometry.PointCloud()
    PCD.points = o3d.utility.Vector3dVector(PointXYZ[:, :3])
    colors = np.hstack([PointXYZ[:, 3:4], PointXYZ[:, 3:4], PointXYZ[:, 3:4]])
    PCD.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])  # 添加坐标系
    o3d.io.write_point_cloud('./image/3d.pcd',PCD)
    # o3d.io.write_image('3d.jpg',PCD)
    # o3d.visualization.draw_geometries([PCD, mesh_frame])

    # 二维显示
    step_size = int(PointXYZ.shape[0] / 100000)
    x_down = PointXYZ[::step_size, 0]
    y_down = PointXYZ[::step_size, 1]
    z_down = PointXYZ[::step_size, 2]
    plt.figure('点云投影')
    # plt.subplot(2, 1, 1)
    plt.plot(x_down, y_down,'b.', label='original values',linewidth=0.1)
    plt.xlabel('x axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title('点云俯视图')
    plt.savefig('image/point_cloud_look_down.jpg')
    plt.close()

    # plt.show()
    # plt.ion()
    # time.sleep(2)
    plt.figure('点云投影')
    plt.plot(x_down, z_down,'b.', label='original values',linewidth=0.1)
    plt.xlabel('x axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title('点云侧视图')
    plt.savefig('image/point_cloud_side_look.jpg')
    plt.close()
    # plt.show()
    # time.sleep(2)
    # print('根据图片修改SKN和FYX等')
    '----------------------------------------4.轴线拟合----------------------------------------'
    print('《《《开始配置轴线要素》》》')
    # dy, dz, xyzforfit = Axis_fitting_VTsShan(PointXYZ, skn_range_yx, skn_range_zx, fyx=fyx, fzx=fzx,LR=LR)  # 二衬计算
    # if dy == 'nan' or dz == 'nan':
    dy, dz, xyzforfit = Axis_fitting_VTsShan(PointXYZ, skn_range_yx, skn_range_zx, fyx=2, fzx=1,LR=LR)  # 二衬计算

    # else:
    #     dy = np.array(list(map(float, dy.split(","))))
    #     dz = np.array(list(map(float, dz.split(","))))


    '----------------------------------------5.拟合线与点云的投影关系----------------------------------------'
    pz = np.poly1d(dz)
    z_poly1d = pz(x_down)  # 该步骤和np.polyval(z1, x_down)效果相同
    py = np.poly1d(dy)
    y_poly1d = py(x_down)  # 该步骤和np.polyval(y1, x_down)效果相同
    plt.close()
    plt.figure('轴线拟合关系')
    # plt.subplot(2, 1, 1)
    # plot_2d(x_down, y_down, title=' y-polyfitting', point_type='b.', label='original values')
    plt.plot(x_down, y_down,'b.', label='original values',linewidth=0.1)
    plt.plot(x_down, y_poly1d,'r.', label='polyfit values',linewidth=0.1)
    if 'xyzforfit' in locals():
        plt.plot(xyzforfit[:, 0], xyzforfit[:, 1], 'g.', label='mean values')
    plt.xlabel('x axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title('点云俯视图--轴线拟合')
    plt.savefig('image/point_cloud_look_down_line.jpg')
    plt.close()

    plt.figure('轴线拟合关系')
    # plt.subplot(2, 1, 2)
    plt.plot(x_down, z_down, 'b.', label='original values')
    plt.plot(x_down, z_poly1d,'r.', label='polyfit values')
    if 'xyzforfit' in locals():
        plt.plot(xyzforfit[:, 0], xyzforfit[:, 2], 'g.', label='mean values')
    plt.xlabel('x axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title('点云侧视图--轴线拟合')
    plt.savefig('image/point_cloud_side_look_line.jpg')
    plt.close()
    # 如果检测没有问题，config对象 DY DZ 写入
    # config.set('self_config', 'dy', str(dy.tolist())[1:-1])  # 给分组设置值
    # config.set('self_config', 'dz', str(dz.tolist())[1:-1])  # 给分组设置值
    # o = open(ini_Filename, 'w')
    # config.write(o)
    # o.close()

    print('dy:',dy,type(dy))
    print('dz',dz,type(dz))
    '----------------------------------------主方向坐标转换----------------------------------------'
    # todo 加入曲线坐标转换
    # 直线坐标转换
    PointXYZ_Trans = np.hstack(
        [PointXYZ[:, 0:1], PointXYZ[:, 1:2] - dy[1], PointXYZ[:, 2:3] - dz[1] + ori_high, PointXYZ[:, 3:4]])
    angle_dy = np.arctan(dy[0])
    angle_dz = np.arctan(dz[0])
    Rz = np.array([[np.cos(angle_dy), np.sin(angle_dy), 0],
                   [-np.sin(angle_dy), np.cos(angle_dy), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(angle_dz), 0, np.sin(angle_dz)],
                   [0, 1, 0],
                   [-np.sin(angle_dz), 0, np.cos(angle_dz)]])
    R = np.dot(Ry, Rz)  # 先转z再y
    T = np.zeros([3, 1])
    PointXYZ_Trans[:, :3] = Coord_Trans(PointXYZ_Trans[:, :3], R, T, 1)
    PointXYZ_Trans = PointXYZ_Trans[PointXYZ_Trans[:, 2] >= 0, :]
    '----------------------------------------6.超欠挖计算----------------------------------------'
    SectElem, OffsetVec = readSectionElem(Para_FileName, 'design')
    print('《《《开始超欠挖计算》》》')
    overBreakValue, azimuthAll = OverBreak(PointXYZ_Trans, dy, dz, skn_range, SectElem, resolution=resolution,
                                           deltH=deltH, delt_inner_radius=delt_inner_radius,
                                           is_test=False, StdLine=StdLine)  # 断面平移高度-2.5m
    # jgAll数据顺序（x,y,z, 方向角 ,超欠挖值  ,f）
    azimuthAll = azimuthAll.reshape(-1, 1)
    jgAll = np.hstack([PointXYZ_Trans[:, :3], azimuthAll, overBreakValue, PointXYZ_Trans[:, -1:]])
    jgAll = jgAll[jgAll[:, -2] > delt_inner_radius, :]  # 去除内轮廓内点云
    # jg数据顺序（x,y,z,θ,超欠挖值,f）

    '----------------------------------------断面切片检查----------------------------------------'

    avg_len = (np.max(jgAll[:, 0]) - np.min(jgAll[:, 0]))/6
    ii = 1
    for i in np.arange(np.min(jgAll[:, 0]), np.max(jgAll[:, 0]), avg_len):
        print(i)
        f2 = plt.figure('第' + str(round(i, 3)) + 'm断面切片')
        xy = tunnelSectionDesign(SectElem, 0, 0)
        plt.plot(xy[0, :], xy[1, :], c='g', label='设计断面')
        # plt.plot(jgAll[:, 0], jgAll[:, 1], c='b')
        point_plt = jgAll[(jgAll[:, 0] > (i - resolution / 2)) & (jgAll[:, 0] < (i + resolution / 2)), :]
        plt.scatter(point_plt[:, 1], point_plt[:, 2], s=0.8, c=point_plt[:, 4], marker='.', label='测量断面点云')
        plt.axis('equal')
        plt.savefig('image/section_%d.jpg'%ii)
        plt.close()
        ii+=1
    '----------------------------------------三维展示检查----------------------------------------'

    f1 = plt.figure('三维展示图', figsize=(10, 6))
    if jgAll.shape[0] == 0:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.text(0.5, 0.5, s='点云数为0或其他错误', ha='center', va='center')
    else:
        jgAll_shape = round(jgAll.shape[0] / 5000)
        ax = plt.axes(projection='3d')
        ax.scatter3D(jgAll[::jgAll_shape, 0], jgAll[::jgAll_shape, 1], jgAll[::jgAll_shape, 2],
                     c=jgAll[::jgAll_shape, 4], marker='.')
        max_len = np.nanmax(jgAll[:, 0]) - np.nanmin(jgAll[:, 0])
        ax.set(xlabel='X',
               ylabel='Y',
               zlabel='Z',
               xticks=np.arange(np.nanmin(jgAll[:, 0]), np.nanmax(jgAll[:, 0]), 5),
               yticks=np.arange(np.nanmin(jgAll[:, 1]) - max_len / 2, np.nanmax(jgAll[:, 1]) + max_len / 2, 5),
               zticks=np.arange(np.nanmin(jgAll[:, 2]) - max_len / 2, np.nanmax(jgAll[:, 2]) + max_len / 2, 5)
               )  # 设置坐标轴格式
    plt.savefig('image/3d_show.jpg')
    print('测试总共用时%.2f秒' % (time.time() - start_all))
    return PointXYZ,x_down,y_down,z_down,ini_Filename,Para_FileName,jgAll

def main():
    '________________________________获取所有原始文件的路径和文件名___________________________________'
    print('《《《开始读取文件》》》')
    # compute_filepath = sys.argv[1] #手动输入文件名
    compute_filepath = '3GGDJAR00100141_20221130_100511.npy'
    horizontal_value = '4.2,16.56'
    '________________________________所有仪器的最新日期的文件循环计算并保存___________________________________'
    # 控制台日志打印
    class Logger(object):
        def __init__(self, filename='default.log', stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

        def close(self):
            self.log.close()

    # 控制台保存日志
    log_filepath = compute_filepath[:compute_filepath.rfind('.npy') + 1]
    logTime = time.strftime("%Y%m%d-%H%M%S") #可能是造成没有日志的原因
    sys.stdout = Logger(log_filepath + logTime + '.log', sys.stdout)
    errLogFn = log_filepath + logTime + '.log_file'
    sys.stderr = Logger(errLogFn, sys.stderr)
    print('Compute_FileName is : %s' % compute_filepath)

    # 要素配置
    # test_config(compute_filepath,horizontal_value)

    if os.path.getsize(errLogFn) == 0:
        Logger.close(sys.stderr)
        os.remove(errLogFn)

if __name__ == '__main__':
    PointXYZ,x_down,y_down,z_down,ini_Filename,Para_FileName = plot_all('E:/WORKFILES/lianzhikeji/plot_qt/3GGDJAR00100141_20221130_100511.npy','4.2,16.56')
    # x_down = list(x_down)
    # y_down = list(y_down)
    # print(y_down.index(min(y_down)))
    # plt.figure('点云投影')
    # # plt.subplot(2, 1, 1)
    # plt.plot(x_down, y_down, 'b.', label='original values', linewidth=0.1)
    # plt.plot([x_down[y_down.index(min(y_down))],x_down[-1]],[y_down[y_down.index(min(y_down))],y_down[-1]],'-.',color='r')
    # plt.xlabel('x axis')
    # # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    # plt.title('点云俯视图')
    # pos = plt.ginput(2)
    # look_down_coor = [x[0] for x in pos]
    # plt.close()
    # plt.figure('点云投影')
    # # plt.subplot(2, 1, 1)
    # plt.plot(x_down, z_down, 'b.', label='original values', linewidth=0.1)
    # plt.xlabel('x axis')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    # plt.title('点云侧视图')
    # pos = plt.ginput(2)
    # look_side_coor = [x[0] for x in pos]
    # plt.close()
    # pointXYZ,dy,dz = check_plot('3GGDJAR00100141_20221130_100511.npy','4.2,16.56',look_down_coor,look_side_coor,'L')
    # print(type(dy),type(dz))
    # plot_section(pointXYZ,'config.ini','parameter.xlsx',dy,dz)
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import sys,os
from tunnel_primary_lining_VTaShan_PreV2 import *
import matplotlib.pyplot as plt
import vtkmodules
from vtkmodules.util.numpy_support import numpy_to_vtk
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# import open3d as o3d



class Ui_Form(QtWidgets.QMainWindow,QtWidgets.QWidget):
    def __init__(self):
        super(Ui_Form, self).__init__()
        QtWidgets.QWidget.__init__(self)
        self.fileName = ''
        self.horizontalValue = ''
        self.origin_pointXYZ = None
        self.ini_Filename = None
        self.Para_FileName = None
        self.x_down = None
        self.y_down = None
        self.z_down = None
        self.look_down_coor = None
        self.look_side_coor = None
        self.direction = None
        self.check_pointXYZ = None
        self.dy = None
        self.dz = None

    def setupUi(self,MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1140, 816)
        MainWindow.setAnimated(False)
        # self.setWindowIcon(QIcon('./image/'))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(820, 0, 80, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.calculating)
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(600, 0, 80, 41))
        self.toolButton.setObjectName("toolButton")
        self.toolButton.clicked.connect(self.file_choose)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(690,0,120,41))
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(0, 0, 591, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.pushButton.setFont(font)

        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(0, 42, 400, 380))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QGridLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        # 添加点云图
        # self.formLayout = QtWidgets.QFormLayout(self.gridLayoutWidget)
        self.frame = QtWidgets.QFrame()
        # vtk.qt.QVTKRenderWindowInteractor()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        # self.formLayout.addWidget(self.vtkWidget)
        self.formLayout.addWidget(self.vtkWidget, 0, 0, 0, 0)

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(410, 40, 600, 380))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        # self.label_2.setPixmap(QtGui.QPixmap('./image/point_cloud_side_look.jpg'))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

        self.label_2.setScaledContents(True)
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        # self.label_3.setPixmap(QtGui.QPixmap('./image/point_cloud_side_look_line.jpg'))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_3.setScaledContents(True)
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)


        # self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        # self.label.setText("")
        # self.label.setObjectName("label")
        # self.gridLayout.addWidget(self.label, 0, 0, 2, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        # self.label_4.setPixmap(QtGui.QPixmap('./image/point_cloud_look_down.jpg'))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.label_4.setScaledContents(True)
        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        # self.label_5.setPixmap(QtGui.QPixmap('./image/point_cloud_look_down_line.jpg'))
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_5.setScaledContents(True)
        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 430, 1000, 341))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        # self.label_7.setPixmap(QtGui.QPixmap('./image/section_5.jpg'))
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.label_7.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label_7, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        # self.label_8.setPixmap(QtGui.QPixmap('./image/section_6.jpg'))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.label_8.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label_8, 1, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        # self.label_6.setPixmap(QtGui.QPixmap('./image/section_4.jpg'))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_6.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label_6, 1, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        # self.label_10.setPixmap(QtGui.QPixmap('./image/section_1.jpg'))
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.label_10.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        # self.label_9.setPixmap(QtGui.QPixmap('./image/3d_show.jpg'))
        self.label_9.setText("")
        self.label_9.setScaledContents(True)
        self.label_9.setObjectName("label_9")
        self.label_9.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label_9, 0, 3, 2, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        # self.label_11.setPixmap(QtGui.QPixmap('./image/section_2.jpg'))
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_11.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label_11, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        # self.label_12.setPixmap(QtGui.QPixmap('./image/section_3.jpg'))
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.label_12.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label_12, 0, 2, 1, 1)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(1010, 100, 121, 531))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.pushButton_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setFont(font)
        self.verticalLayout.addWidget(self.pushButton_5)
        self.pushButton_5.clicked.connect(self.plot_clear)
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.comboBox.addItem("请选择方向")
        self.comboBox.addItem("L")
        self.comboBox.addItem("R")
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("combobox")
        self.verticalLayout.addWidget(self.comboBox)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setFont(font)
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_2.clicked.connect(self.get_look_down_pos)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setFont(font)
        self.verticalLayout.addWidget(self.pushButton_3)
        self.pushButton_3.clicked.connect(self.get_look_side_pos)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.setFont(font)
        self.verticalLayout.addWidget(self.pushButton_6)
        self.pushButton_6.clicked.connect(self.check_line)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.setFont(font)
        self.verticalLayout.addWidget(self.pushButton_7)
        self.pushButton_7.clicked.connect(self.section_plot)
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setFont(font)
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_4.clicked.connect(self.check_correct)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1140, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "配置表修改工具"))
        self.pushButton.setText(_translate("MainWindow", "开始计算"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.lineEdit.setToolTip(_translate("MainWindow", "请请选择文件"))
        self.lineEdit.setText(_translate("MainWindow", "请选择文件"))
        # self.lineEdit_2.setText(_translate("MainWindow","选择方向"))
        self.lineEdit_4.setText(_translate("MainWindow","输入倾角"))
        self.pushButton_2.setText(_translate("MainWindow", "俯视图选点"))
        self.pushButton_3.setText(_translate("MainWindow", "侧视图选点"))
        self.pushButton_4.setText(_translate("MainWindow", "校验正确"))
        self.pushButton_5.setText(_translate("MainWindow","重新校验"))
        self.pushButton_6.setText(_translate("MainWindow", "开始校验"))
        self.pushButton_7.setText(_translate("MainWindow", "截面3D图"))


    def file_choose(self):
        try:
            fileName = QtWidgets.QFileDialog.getOpenFileNames(self,'choose_file',os.getcwd())
            self.fileName = fileName[0][0]
            self.lineEdit.setText(fileName[0][0])
        except Exception as e:
            pass

    def calculating(self):
        self.horizontalValue = self.lineEdit_4.text()
        if self.fileName and ',' in self.horizontalValue:
            if self.fileName.endswith('npy'):
                self.pushButton.setText("计算中")
                time.sleep(1)
                try:
                    print(self.fileName)
                    self.pointXYZ,self.x_down,self.y_down,self.z_down,self.ini_Filename,self.Para_FileName = plot_all(self.fileName,self.horizontalValue)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, '错误', '计算出错，错误为:%s'%e.__str__())
                    self.pushButton.setText("重新计算")
                    return
                self.label_2.setPixmap(QtGui.QPixmap('./image/point_cloud_look_down_line.jpg'))
                self.label_3.setPixmap(QtGui.QPixmap('./image/point_cloud_side_look_line.jpg'))
                self.label_4.setPixmap(QtGui.QPixmap('./image/point_cloud_look_down.jpg'))
                self.label_5.setPixmap(QtGui.QPixmap('./image/point_cloud_side_look.jpg'))
                self.label_7.setPixmap(QtGui.QPixmap('./image/section_5.jpg'))
                self.label_8.setPixmap(QtGui.QPixmap('./image/section_6.jpg'))
                self.label_6.setPixmap(QtGui.QPixmap('./image/section_4.jpg'))
                self.label_10.setPixmap(QtGui.QPixmap('./image/section_1.jpg'))
                self.label_9.setPixmap(QtGui.QPixmap('./image/3d_show.jpg'))
                self.label_11.setPixmap(QtGui.QPixmap('./image/section_2.jpg'))
                self.label_12.setPixmap(QtGui.QPixmap('./image/section_3.jpg'))
                self.ren = vtk.vtkRenderer()
                self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
                self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

                # pcd = np.load(self.fileName)
                pcd = self.get_np_pcd()
                # pcd = o3d.io.read_point_cloud("./image/3d.pcd")
                # 新建 vtkPoints 实例
                points = vtk.vtkPoints()
                # 导入点数据
                points.SetData(numpy_to_vtk(pcd[:,:3]))
                # 新建 vtkPolyData 实例
                polydata = vtk.vtkPolyData()
                # 设置点坐标
                polydata.SetPoints(points)
                # 顶点相关的 filter
                vertex = vtk.vtkVertexGlyphFilter()
                vertex.SetInputData(polydata)
                # mapper 实例
                mapper = vtk.vtkPolyDataMapper()
                # 关联 filter 输出
                mapper.SetInputConnection(vertex.GetOutputPort())
                # actor 实例
                actor = vtk.vtkActor()
                # 关联 mapper
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0, 1, 0)
                transform = vtk.vtkTransform()
                transform.Translate(1.0, 0, 0)
                axes = vtk.vtkAxesActor()
                axes.SetUserTransform(transform)
                axes.SetTotalLength(5, 5, 5)
                self.ren.SetBackground(0, 0, 0)
                self.ren.AddActor(axes)
                self.ren.AddActor(actor)
                self.ren.ResetCamera()
                self.iren.Initialize()
                self.pushButton.setText("计算完成")
            else:
                QtWidgets.QMessageBox.warning(self,'错误','文件选择错误，请选择以.npy结尾的文件')
        elif ',' not in self.horizontalValue:
            QtWidgets.QMessageBox.warning(self, '错误', '倾角不正确，请以逗号分隔倾角值')


    def plot_clear(self):
        self.label_2.hide()
        self.label_3.hide()
        self.label_7.hide()
        self.label_8.hide()
        self.label_6.hide()
        self.label_10.hide()
        self.label_9.hide()
        self.label_11.hide()
        self.label_12.hide()
        if os.path.exists('./image/point_cloud_look_down_line.jpg'):
            os.remove('./image/point_cloud_look_down_line.jpg')
        if os.path.exists('./image/point_cloud_side_look_line.jpg'):
            os.remove('./image/point_cloud_side_look_line.jpg')
        if os.path.exists('./image/section_1.jpg'):
            os.remove('./image/section_1.jpg')
        if os.path.exists('./image/section_2.jpg'):
            os.remove('./image/section_2.jpg')
        if os.path.exists('./image/section_3.jpg'):
            os.remove('./image/section_3.jpg')
        if os.path.exists('./image/section_4.jpg'):
            os.remove('./image/section_4.jpg')
        if os.path.exists('./image/section_5.jpg'):
            os.remove('./image/section_5.jpg')
        if os.path.exists('./image/section_6.jpg'):
            os.remove('./image/section_6.jpg')
        if os.path.exists('./image/3d_show.jpg'):
            os.remove('./image/3d_show.jpg')


    def get_look_down_pos(self):
        self.direction = self.comboBox.currentText()
        if self.direction == 'L':
            x_down = list(self.x_down)
            y_down = list(self.y_down)
            plt.figure('点云投影')
            plt.plot(self.x_down, self.y_down, 'b.', label='original values', linewidth=0.1)
            plt.plot([x_down[0],x_down[y_down.index(max(y_down))]],[y_down[0],y_down[y_down.index(max(y_down))]],'-.',color='r')
            plt.xlabel('x axis')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
            plt.title('点云俯视图')
            pos = plt.ginput(2)
            self.look_down_coor = [x[0] for x in pos]
            plt.close()
        elif self.direction == 'R':
            x_down = list(self.x_down)
            y_down = list(self.y_down)
            plt.figure('点云投影')
            plt.plot(self.x_down, self.y_down, 'b.', label='original values', linewidth=0.1)
            plt.plot([x_down[y_down.index(min(y_down))],x_down[-1]],[y_down[y_down.index(min(y_down))],y_down[-1]],'-.',color='r')
            plt.xlabel('x axis')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
            plt.title('点云俯视图')
            pos = plt.ginput(2)
            self.look_down_coor = [x[0] for x in pos]
            plt.close()
        else:
            QtWidgets.QMessageBox.warning(self, '错误', '请选择方向')


    def get_look_side_pos(self):
        plt.figure('点云投影')
        plt.plot(self.x_down, self.z_down, 'b.', label='original values', linewidth=0.1)
        plt.xlabel('x axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
        plt.title('点云侧视图')
        pos = plt.ginput(2)
        self.look_side_coor = [x[0] for x in pos]
        plt.close()

    def check_line(self):
        self.direction = self.comboBox.currentText()
        try:
            self.check_pointXYZ,self.dy,self.dz= check_plot(self.fileName,self.horizontalValue,self.look_down_coor,self.look_side_coor,self.direction,self.ini_Filename,self.Para_FileName)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, '错误', '校验失败，请重新选择坐标点！错误为：%s'%e.__str__())
        self.label_2.show()
        self.label_2.setPixmap(QtGui.QPixmap('./image/point_cloud_look_down_line.jpg'))
        self.label_3.show()
        self.label_3.setPixmap(QtGui.QPixmap('./image/point_cloud_side_look_line.jpg'))


    def section_plot(self):
        # print(self.ini_Filename,self.Para_FileName,type(self.dy),type(self.dz))
        try:
            plot_section(self.check_pointXYZ,self.ini_Filename,self.Para_FileName,self.dy,self.dz)
        except Exception as e:
            print(e)
        self.label_7.show()
        self.label_8.show()
        self.label_6.show()
        self.label_10.show()
        self.label_9.show()
        self.label_11.show()
        self.label_12.show()
        self.label_7.setPixmap(QtGui.QPixmap('./image/section_5.jpg'))
        self.label_8.setPixmap(QtGui.QPixmap('./image/section_6.jpg'))
        self.label_6.setPixmap(QtGui.QPixmap('./image/section_4.jpg'))
        self.label_10.setPixmap(QtGui.QPixmap('./image/section_1.jpg'))
        self.label_9.setPixmap(QtGui.QPixmap('./image/3d_show.jpg'))
        self.label_11.setPixmap(QtGui.QPixmap('./image/section_2.jpg'))
        self.label_12.setPixmap(QtGui.QPixmap('./image/section_3.jpg'))

    def check_correct(self):
        if modfy_config(self.dy,self.dz,self.ini_Filename):
            QtWidgets.QMessageBox.information(self, '消息', '修改成功！！')
            self.label_2.hide()
            self.label_3.hide()
            self.label_4.hide()
            self.label_5.hide()
            self.label_7.hide()
            self.label_8.hide()
            self.label_6.hide()
            self.label_10.hide()
            self.label_9.hide()
            self.label_11.hide()
            self.label_12.hide()
            if os.path.exists('./image/point_cloud_look_down.jpg'):
                os.remove('./image/point_cloud_look_down.jpg')
            if os.path.exists('./image/point_cloud_side_look.jpg'):
                os.remove('./image/point_cloud_side_look.jpg')
            if os.path.exists('./image/point_cloud_look_down_line.jpg'):
                os.remove('./image/point_cloud_look_down_line.jpg')
            if os.path.exists('./image/point_cloud_side_look_line.jpg'):
                os.remove('./image/point_cloud_side_look_line.jpg')
            if os.path.exists('./image/section_1.jpg'):
                os.remove('./image/section_1.jpg')
            if os.path.exists('./image/section_2.jpg'):
                os.remove('./image/section_2.jpg')
            if os.path.exists('./image/section_3.jpg'):
                os.remove('./image/section_3.jpg')
            if os.path.exists('./image/section_4.jpg'):
                os.remove('./image/section_4.jpg')
            if os.path.exists('./image/section_5.jpg'):
                os.remove('./image/section_5.jpg')
            if os.path.exists('./image/section_6.jpg'):
                os.remove('./image/section_6.jpg')
            if os.path.exists('./image/3d_show.jpg'):
                os.remove('./image/3d_show.jpg')
            if os.path.exists('./image/3d.pcd'):
                os.remove('./image/3d.pcd')
        else:
            QtWidgets.QMessageBox.warning(self, '错误', '修改失败，请检查文件！')


    def get_np_pcd(self):
        config = configparser.ConfigParser()
        # -read读取ini文件
        config.clear()
        for encoding in 'GB18030', 'UTF-8-sig':
            try:
                config.read(self.ini_Filename, encoding=encoding)
            except:
                continue
        skn_range = list(map(float, config.get('self_config', 'skn_range').split(",")))
        skn_range_yx = list(map(float, config.get('self_config', 'skn_range_yx').split(",")))
        skn_range_zx = list(map(float, config.get('self_config', 'skn_range_zx').split(",")))
        horizontalValue = self.horizontalValue.split(',')
        maxRange = max(max(skn_range), max(skn_range_yx), max(skn_range_zx))  # 单位m
        PointXYZ = read_data(self.fileName, maxRange=maxRange, if_unique=True)[:, :4]
        PointXYZ = PointXYZ.astype('float64')
        PointXYZ[:, :3] = PointXYZ[:, :3] / 1000
        PointXYZ[:, 3] = PointXYZ[:, 3] / 50  # 反射率以由1-100转换为0-1
        # 旋转
        gamma = np.pi
        R = np.array([[1, 0, 0],
                      [0, np.cos(gamma), -np.sin(gamma)],
                      [0, np.sin(gamma), np.cos(gamma)]])
        T = np.zeros([3, 1])
        PointXYZ[:, :3] = Coord_Trans(PointXYZ[:, :3], R, T, 1)
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
        return PointXYZ


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
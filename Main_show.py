import sys
from imutils.video import VideoStream
from imutils.video import FPS
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import tkinter as tk
from tkinter import filedialog
from tkinter.constants import BOTH, LEFT, RIGHT, TOP
from tkinter.filedialog import *
from tkinter import ttk
import cv2
import numpy as np
from collections import deque
import argparse
import imutils
import time
import predict
import threading
from PIL import Image, ImageTk


class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green": ("绿牌", "#55FF55"), "yello": (
        "黄牌", "#FFFF00"), "blue": ("蓝牌", "#6666FF")}

    def __init__(self, win):  #通过tkinter创建车牌识别系统界面
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("车牌识别系统")
        win.geometry('800x600')
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        ttk.Label(frame_left, text='检测车牌图片：').pack(anchor="nw")
        ttk.Label(frame_right1, text='车牌区域展示：').grid(
            column=0, row=0, sticky=tk.W)

        from_pic_ctl = ttk.Button(
            frame_right2, text="来自图片的车牌", width=20, command=self.from_pic)
        from_vedio_ctl = ttk.Button(
            frame_right2, text="来自摄像头的车牌", width=20, command=self.from_vedio)
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='车牌识别结果：').grid(
            column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_vedio_ctl.pack(anchor="se", pady="5")
        from_pic_ctl.pack(anchor="se", pady="5")
        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)  # array转换成image
        imgtk = ImageTk.PhotoImage(image=im)  # 一个与tkinter兼容的照片图像
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)

            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def show_roi(self, r, roi, color):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.update_time = time.time()
            try:
                c = self.color_transform[color]
                self.color_ctl.configure(
                    text=c[0], background=c[1], state='enable')
            except:
                self.color_ctl.configure(state='disabled')
        elif self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            self.color_ctl.configure(state='disabled')


    def from_vedio(self, mBox=None):
        if self.thread_run:
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                mBox.showwarning('警告', '摄像头打开失败！')
                self.camera = None
                return
        self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True


    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
        if self.pic_path:
            img_bgr = predict.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
            for resize_rate in resize_rates:
                try:
                    r, roi, color = self.predictor.predict(img_bgr, resize_rate)
                except:
                    continue
                if r:
                    break
            self.show_roi(r, roi, color)
            
    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            if time.time() - predict_time > 2:
                r, roi, color = self.predictor.predict(img_bgr)
                self.show_roi(r, roi, color)
                predict_time = time.time()
        print("run end")
        

def resize_keep_aspectratio(image_src, dst_size): #对图片进行等比例缩放，在边界框时将图片缩放到合适的尺寸。
    src_h, src_w = image_src.shape[:2]
    # print(src_h, src_w)
    dst_h, dst_w = dst_size

    # 判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))

    h_, w_ = image_dst.shape[:2]

    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)

    value = [0, 0, 0]
    borderType = cv2.BORDER_CONSTANT
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)
    return image_dst

class Ui_MainWindow(object): #主界面的设定（运用实验课程的架使用Tkinter模块配合QT进行主页面的设计）
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    image_src = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    dst_size = (720, 720)
    image = resize_keep_aspectratio(image_src, dst_size)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.image.shape[1]+180, self.image.shape[0]+100)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(80, 60, self.image.shape[1],self.image.shape[0]))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        font = QtGui.QFont()
        font.setKerning(True)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1145, 26))
        self.menubar.setFont(font)
        self.menubar.setMouseTracking(True)
        self.menubar.setTabletTracking(False)
        self.menubar.setAcceptDrops(True)
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setPriority(QtWidgets.QAction.HighPriority)
        self.action.setObjectName("action")
        self.menu.addAction(self.action)
        self.menubar.addAction(self.menu.menuAction())
        self.pushButton_1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_1.setObjectName("pushButton_1")
        self.horizontalLayout.addWidget(self.pushButton_1)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)

        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.pushButton_1.clicked.connect(self.btn1_clicked)
        self.pushButton_2.clicked.connect(self.btn2_clicked)
        self.pushButton_3.clicked.connect(self.btn3_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "唐一铭车牌识别系统主页面"))
        self.action.setText(_translate("MainWindow", "打开图片"))
        self.action.setStatusTip(_translate("MainWindow", "打开图片"))
        self.action.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.pushButton_1.setText(_translate("MainWindow", "打开原图"))
        self.pushButton_2.setText(_translate("MainWindow", "增加噪声"))
        self.pushButton_3.setText(_translate("MainWindow", "车牌检测"))
        self.label.setText(_translate("MainWindow", "唐一铭车牌检测系统"))


    def btn1_clicked(self):
        self.image_1 = self.image
        self.image_1 = QtGui.QImage(self.image_1.data, self.image_1.shape[1], self.image_1.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_1))

    def btn2_clicked(self):
        self.img_noise=np.zeros(self.image.shape,dtype=np.uint8)
        thred=0.1
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                ratio=np.random.rand()
                if ratio<thred:
                    self.img_noise[i,j]=255
                else:
                    self.img_noise[i,j]=self.image[i,j]
        self.img_noise=QtGui.QImage(self.img_noise.data,self.img_noise.shape[1],self.img_noise.shape[0],QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.img_noise))



    def btn3_clicked(self):  
        def close_window():
            print("destroy")
            if surface.thread_run :
                surface.thread_run = False
                surface.thread.join(2.0)
            win.destroy()

        win=tk.Toplevel() 
        surface = Surface(win)
        win.protocol('WM_DELETE_WINDOW', close_window)
        win.mainloop()
        sys.exit(app.exec_())
   

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
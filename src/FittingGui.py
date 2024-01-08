import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit,QLabel,QRadioButton,QTextEdit,QMessageBox,QFileDialog,QTabWidget,QGridLayout,QComboBox,QSizePolicy,QSplashScreen
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
from scipy.optimize import curve_fit 
import pandas as pd
import logging
import time

# 현재 버전
CURRENT_VERSION = '1.0.2'

# 가우시안 함수 정의
def gaussian(x, amplitude, mean, stddev , y0):
    return y0 + amplitude * np.exp(-((x - mean) ** 2) / (2 * (stddev ** 2)))



class DataFittngApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_path = None
        self.data = None
        self.pixel_size = 3.45
        self.magnification = 1
        self.saturation_limit = 0
        self.file_name = None
        self.amp = 0
        self.mean = 0
        self.std = 0
        self.y0 = 0
        self.fwhm = 0
        self.e2 = 0
        self.is_draw = False
        self.is_fit = False
        self.small_size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.text_chage = False
        self._init_ui()

    
    def _init_ui(self):
        # 메인 위젯 설정
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # 메인 레이아웃 설정
        self.layout = QHBoxLayout(self.main_widget)

  

        # 그래프 캔버스 설정
        self.canvas_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas_layout.addWidget(self.canvas)
        self.layout.addLayout(self.canvas_layout)




        # 컨트롤 패널 설정
        self.control_panel = QWidget(self)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.layout.addWidget(self.control_panel)

        # 이미지 삽입
        self.image_layout = QHBoxLayout()
        self.image_label = QLabel(self.control_panel)
        self.image_label.setPixmap(QPixmap(resource_path('assets/gaussian.png')))
        # 스타일 변경
        self.image_label.setScaledContents(True)
        self.image_layout.minimumSize()
        self.image_layout.addStretch(1)
        self.image_layout.addWidget(self.image_label)
        self.image_layout.addStretch(1)
        self.control_layout.addLayout(self.image_layout)

        
        # data_path 설정
        self.data_path_layout = QHBoxLayout()
        self.data_path_label = QLabel("Data Path",self.control_panel)
        self.data_path_line = QLineEdit(self.control_panel)
        self.data_path_line.textChanged.connect(self.update_data_path)
        self.data_path_button = QPushButton("Open",self.control_panel)
        self.data_path_button.clicked.connect(self.open_data_folder)
        self.data_path_layout.addWidget(self.data_path_label)
        self.data_path_layout.addWidget(self.data_path_line)
        self.data_path_layout.addWidget(self.data_path_button)
        self.control_layout.addLayout(self.data_path_layout)

        # pixel size, magnification saturation_limit 설정
        self.pixel_size_and_magnification_layout = QHBoxLayout()
        self.pixel_size_label = QLabel("Pixel Size",self.control_panel)
        self.pixel_size_line = QLineEdit(self.control_panel)
        self.pixel_size_line.setPlaceholderText("um")
        self.pixel_size_line.setText(f"{self.pixel_size}")
        self.pixel_size_line.textChanged.connect(self.update_pixel_size)
        self.magnification_label = QLabel("Magnification",self.control_panel)
        self.magnification_line = QLineEdit(self.control_panel)
        self.magnification_line.setPlaceholderText("X")
        self.magnification_line.setText(f"{self.magnification}")
        self.magnification_line.textChanged.connect(self.update_magnification)
        self.saturation_limit_label = QLabel("Saturation Limit",self.control_panel)
        self.saturation_limit_line = QLineEdit(self.control_panel)
        self.saturation_limit_line.setText(f'{self.saturation_limit}')
        self.saturation_limit_line.textChanged.connect(self.update_saturation_limit)
        self.pixel_size_and_magnification_layout.addWidget(self.pixel_size_label)
        self.pixel_size_and_magnification_layout.addWidget(self.pixel_size_line)
        self.pixel_size_and_magnification_layout.addWidget(self.magnification_label)
        self.pixel_size_and_magnification_layout.addWidget(self.magnification_line)
        self.pixel_size_and_magnification_layout.addWidget(self.saturation_limit_label)
        self.pixel_size_and_magnification_layout.addWidget(self.saturation_limit_line)
        self.control_layout.addLayout(self.pixel_size_and_magnification_layout)

        
         
        # fitting method show
        self.fitting_method_show_layout = QHBoxLayout()

        self.fitting_method_grid = QGridLayout()
        self.fitting_method_grid.setHorizontalSpacing(80)
       
        self.fitting_method_file_name_label = QLabel("File Name :",self.control_panel)
        self.fitting_method_file_name_QComboBox = QComboBox(self.control_panel)
        self.fitting_method_file_name_QComboBox.currentIndexChanged.connect(self.update_item)
        self.fitting_method_fwhm_label = QLabel("FWHM :",self.control_panel)
        self.fitting_method_fwhm_line = QLabel(f'{self.fwhm}',self.control_panel)
        self.fitting_method_w_label = QLabel("1/e^2 :",self.control_panel)
        self.fitting_method_w_line = QLabel(f'{self.e2}',self.control_panel)
        self.fitting_method_grid.addWidget(self.fitting_method_file_name_label,0,0)
        self.fitting_method_grid.addWidget(self.fitting_method_file_name_QComboBox,0,1)
        self.fitting_method_grid.addWidget(self.fitting_method_fwhm_label,1,0)
        self.fitting_method_grid.addWidget(self.fitting_method_fwhm_line,1,1)
        self.fitting_method_grid.addWidget(self.fitting_method_w_label,2,0)
        self.fitting_method_grid.addWidget(self.fitting_method_w_line,2,1)

        # fitting method control
        self.fitting_method_y0_label = QLabel("y0 :",self.control_panel)
        self.fitting_method_y0_line = QLineEdit(self.control_panel)
        self.fitting_method_y0_line.setSizePolicy(self.small_size_policy)
        self.fitting_method_y0_line.setPlaceholderText("y0")
        self.fitting_method_y0_line.setText(f"{self.y0}")
        self.fitting_method_mean_label = QLabel("Xc :",self.control_panel)
        self.fitting_method_mean_line = QLineEdit(self.control_panel)
        self.fitting_method_mean_line.setPlaceholderText("Mean")
        self.fitting_method_mean_line.setText(f"{self.mean}")
        self.fitting_method_std_label = QLabel("w :",self.control_panel)
        self.fitting_method_std_line = QLineEdit(self.control_panel)
        self.fitting_method_std_line.setPlaceholderText("Std")
        self.fitting_method_std_line.setText(f"{self.std}")
        self.fitting_method_amp_label = QLabel("A :",self.control_panel)
        self.fitting_method_amp_line = QLineEdit(self.control_panel)
        self.fitting_method_amp_line.setPlaceholderText("A")
        self.fitting_method_amp_line.setText(f"{self.amp}")

        self.fitting_method_y0_line.textChanged.connect(self.update_y0)
        self.fitting_method_mean_line.textChanged.connect(self.update_mean)
        self.fitting_method_std_line.textChanged.connect(self.update_std)
        self.fitting_method_amp_line.textChanged.connect(self.update_amp)


        self.fitting_method_grid.addWidget(self.fitting_method_y0_label,0,3)
        self.fitting_method_grid.addWidget(self.fitting_method_y0_line,0,4)
        self.fitting_method_grid.addWidget(self.fitting_method_mean_label,1,3)
        self.fitting_method_grid.addWidget(self.fitting_method_mean_line,1,4)
        self.fitting_method_grid.addWidget(self.fitting_method_std_label,2,3)
        self.fitting_method_grid.addWidget(self.fitting_method_std_line,2,4)
        self.fitting_method_grid.addWidget(self.fitting_method_amp_label,3,3)
        self.fitting_method_grid.addWidget(self.fitting_method_amp_line,3,4)
        

        
        
        
        self.fitting_method_show_layout.addLayout(self.fitting_method_grid)


        # fit ,save , auit , help 버튼
        self.fitting_method_button_layout = QVBoxLayout()
        self.fitting_method_help_button = QPushButton("Help",self.control_panel)
        self.fitting_method_help_button.clicked.connect(self.show_help)
        self.fitting_method_fit_button = QPushButton("Fit",self.control_panel)
        self.fitting_method_fit_button.clicked.connect(self.fit)
        self.fitting_method_save_button = QPushButton("Save",self.control_panel)
        self.fitting_method_save_button.clicked.connect(self.save)
        self.fitting_method_quit_button = QPushButton("Quit",self.control_panel)
        self.fitting_method_quit_button.clicked.connect(self.close)
        self.fitting_method_button_layout.addWidget(self.fitting_method_help_button)
        self.fitting_method_button_layout.addWidget(self.fitting_method_fit_button)
        self.fitting_method_button_layout.addWidget(self.fitting_method_save_button)
        self.fitting_method_button_layout.addWidget(self.fitting_method_quit_button)
        self.fitting_method_show_layout.addLayout(self.fitting_method_button_layout)

        


        self.control_layout.addLayout(self.fitting_method_show_layout)




    



        # open 뺴고 전부 잠그기
        self.pixel_size_line.setReadOnly(True)
        self.magnification_line.setReadOnly(True)
        self.saturation_limit_line.setReadOnly(True)
        self.fitting_method_file_name_QComboBox.setDisabled(True)
        self.fitting_method_y0_line.setReadOnly(True)
        self.fitting_method_mean_line.setReadOnly(True)
        self.fitting_method_std_line.setReadOnly(True)
        self.fitting_method_amp_line.setReadOnly(True)
        self.fitting_method_fit_button.setDisabled(True)
        self.fitting_method_save_button.setDisabled(True)
        
        


   

        
        
        # 하단 상태바
        self.statusBar().showMessage('Connect Data path')
        
        # GUI 크기 및 타이틀 설정
        # self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Fitting GUI')
        self.show()
    


############################fitting###########
    def fit(self):
        '''
        saturation을 고려한 가우시안 피팅 함수
        saturation이 생긴 부분의 데이터를 제거하고 피팅을 수행한다.

        data : pandas dataframe
        saturation_limit : saturation이 생기는 y값의 limit
        data_name : 그래프의 이름
        draw : boolean  True : draw graph , False : not draw graph
        
        return :
        popt : fitting parameter  gaussian 함수의 parameter 3개 (amplitude, mean, stddev)
        fwhm : fwhm
        x_full : full data x    피팅 데이터 x
        y_full : full data y    피팅 데이터 y
        x : unsaturated data x  saturation이 생시는 부분을 제거한 데이터
        y : unsaturated data y  saturation이 생시는 부분을 제거한 데이터
        e2 : 1/e^2
        '''
        


        self.statusBar().showMessage("Fitting...")

   


        ######## saturation 예측 피팅 #############3
        unsaturated_data = self.data
        # saturation이 생기는 부분을 제거
        if self.saturation_limit > 0:
            unsaturated_data = self.data[self.data[:, 1] < self.saturation_limit]

        # saturation이 안생긴 데이터를 x와 y로 분리
        self.unsaturated_x = unsaturated_data[:, 0]
        self.unsaturated_y = unsaturated_data[:, 1]
        

        # 가우시안 피팅 수행
        popt, pcov = curve_fit(gaussian, self.unsaturated_x, self.unsaturated_y, p0=[max(self.unsaturated_y), np.mean(self.unsaturated_x), np.std(self.unsaturated_x),0])

        self.amp = popt[0]
        self.mean = popt[1]
        self.std = popt[2]
        self.y0 = popt[3]

        self.updata_fit_graph()
        
        self.fitting_method_amp_line.setReadOnly(False)
        self.fitting_method_mean_line.setReadOnly(False)
        self.fitting_method_std_line.setReadOnly(False)
        self.fitting_method_y0_line.setReadOnly(False)
        
        

   

################################################### REST OF THR FEATURES ########################################################################

    def convert_pixel_to_um(self,pixel):
        '''
        pixel을 um로 변환

        pixel size / magnification = pixel resolution
        pixel * pixel resolution = um

        '''
        return pixel * (float(self.pixel_size_line.text())/ float(self.magnification_line.text()))

    def update_item(self):
        ''''''
        try:
            df = pd.read_excel(os.path.join(self.data_path,self.fitting_method_file_name_QComboBox.currentText()))
            self.data = df.to_numpy()
            self.data[:,0] = self.convert_pixel_to_um(self.data[:,0])
            if self.is_fit:
                self.fit()
            else :self.update_graph(self.data[:,0],self.data[:,1])

        except:
            print("error")
            return

 

    

    # data_path open
    def open_data_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self,"Select Directory")
        self.data_path_line.setText(folder_path)    


    # help
    def show_help(self):
        help_pdf_file = resource_path('assets/LaserMicroscopeGUIMANUAL.pdf')
        os.startfile(help_pdf_file)
        

    # 에러 메시지
    def show_error(self , title = "입력 오류" , sub_title = "알맞은 값을 입력해주세요."):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(sub_title)
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

    # 그래프 업데이트
    def update_graph(self, x, y):
        ''''''
        self.statusBar().showMessage("Drawing...")

        self.is_draw = True
        self.is_fit = False

        self.ax.clear()
        sc = self.ax.plot(x, y , 'black')

        self.canvas.draw()

        self.statusBar().showMessage("Ready")

    def updata_fit_graph(self):
        ''''''
        self.statusBar().showMessage("Drawing...")
        self.is_fit = True
        popt = [self.amp, self.mean, self.std, self.y0]
        # 
        x_full = np.linspace(min(self.data[:, 0]), max(self.data[:, 0]), 5000)
        y_full = gaussian(x_full, *popt)

        # FWHM 계산
        # fwhm = abs(2 * np.sqrt(2 * np.log(2)) * popt[2])
        self.fwhm = abs(2 * np.sqrt( np.log(4)) * self.std)

        # 1/e^2  2w
        self.e2 = 1.699 * self.fwhm



        # FWHM의 시작 및 끝 좌표 계산
        self.fwhm_start_x = self.mean - self.fwhm / 2
        self.fwhm_end_x = self.mean + self.fwhm / 2
        self.fwhm_y = gaussian(self.mean, *popt) / 2  + self.y0

        # 1/e^2의 시작 및 끝 좌표 계산
        self.e2_start_x = self.mean - self.e2 / 2
        self.e2_end_x = self.mean + self.e2 / 2
        self.e2_y = gaussian(self.mean, *popt) / np.exp(2) + self.y0


        self.ax.clear()
        self.ax.plot(self.unsaturated_x, self.unsaturated_y , 'black' , label = 'Origin Data')
        self.ax.plot([self.fwhm_start_x, self.fwhm_end_x], [self.fwhm_y, self.fwhm_y], 'g--', label='FWHM')
        self.ax.plot([self.e2_start_x, self.e2_end_x], [self.e2_y, self.e2_y], 'b--', label='1/e^2')
        self.ax.plot(x_full,y_full, 'r--', label='Gaussian Fit')
        self.ax.set_title(f'{self.fitting_method_file_name_QComboBox.currentText()}\nFWHM = {round(self.fwhm_end_x - self.fwhm_start_x,3)}\n1/e^2 = {round(self.e2_end_x - self.e2_start_x,3)}')
        self.ax.set_xlabel('Distance(um)')
        self.ax.set_ylabel('Intensity')
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

        self.updata_fwhm_e2()

        if not self.text_chage:
            self.fitting_method_amp_line.blockSignals(True)
            self.fitting_method_mean_line.blockSignals(True)
            self.fitting_method_std_line.blockSignals(True)
            self.fitting_method_y0_line.blockSignals(True)

            self.fitting_method_amp_line.setText(f"{round(self.amp,3)}")
            self.fitting_method_mean_line.setText(f"{round(self.mean,3)}")
            self.fitting_method_std_line.setText(f"{round(self.std,3)}")
            self.fitting_method_y0_line.setText(f"{round(self.y0,3)}")

            self.fitting_method_amp_line.blockSignals(False)
            self.fitting_method_mean_line.blockSignals(False)
            self.fitting_method_std_line.blockSignals(False)
            self.fitting_method_y0_line.blockSignals(False)

        self.text_chage = False

        self.statusBar().showMessage("Ready")

    


    def updata_fwhm_e2(self):
        ''''''
        self.fitting_method_fwhm_line.setText(f"{round(self.fwhm,3)} um")
        self.fitting_method_w_line.setText(f"{round(self.e2,3)} um")
 
    

    # save
    def save(self):
        options = QFileDialog.Options()
        file_name_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*);;PNG Files (*.png);;JPEG Files (*.jpg)", options=options)
        if file_name_path:
            self.ax.figure.savefig(file_name_path)
            

        

    def closeEvent(self, event):
        reply =QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            
            event.accept()

        else:
            event.ignore()

###########################button change########################
    def update_data_path(self):
        try:
            os.listdir(self.data_path_line.text())
            self.data_path = self.data_path_line.text()
            file_list = os.listdir(self.data_path) 
            self.fitting_method_file_name_QComboBox.clear()
            for file_name in file_list:
                if file_name.endswith(".xlsx"):
                    self.fitting_method_file_name_QComboBox.addItem(file_name)     

            if self.fitting_method_file_name_QComboBox.count() == 0:
                return
            self.statusBar().showMessage("Ready")

            self.fitting_method_file_name_QComboBox.setDisabled(False)
            self.fitting_method_amp_line.setReadOnly(False)
            self.fitting_method_fit_button.setDisabled(False)
            self.fitting_method_save_button.setDisabled(False)
            self.magnification_line.setReadOnly(False)
            self.pixel_size_line.setReadOnly(False)
            self.saturation_limit_line.setReadOnly(False)

        except:
            return

    def update_pixel_size(self):
        try:
            self.pixel_size = float(self.pixel_size_line.text())
            if not self.is_draw:
                return
            
            self.update_item()
        except:
            return
        
    def update_magnification(self):
        try:
            self.magnification = float(self.magnification_line.text())
            if not self.is_draw:
                return
            
            self.update_item()
        except:
            return

    def update_saturation_limit(self):
        try:
            self.saturation_limit = float(self.saturation_limit_line.text())
    
        except:
            return

    def update_y0(self):
        ''''''
        try:
            self.y0 = float(self.fitting_method_y0_line.text())
            self.text_chage = True
            self.updata_fit_graph()
        except:
            return
        
    def update_mean(self):
        ''''''
        try:
            self.mean = float(self.fitting_method_mean_line.text())
            self.text_chage = True
            self.updata_fit_graph()
        except:
            return
        
    def update_std(self):
        ''''''
        try:
            self.std = float(self.fitting_method_std_line.text())
            self.text_chage = True
            self.updata_fit_graph()
        except:
            return
        
    def update_amp(self):
        ''''''
        try:
            self.amp = float(self.fitting_method_amp_line.text())
            self.text_chage = True
            self.updata_fit_graph()
        except:
            return
        

##########################################################

# 업데이트 버전 체크
def check_version(splash):
    # 버전 체크 로직
    for i in range(1, 4):
        time.sleep(1)  # 실제 버전 체크 로직으로 대체
        splash.showMessage(f"버전 확인 중... {i}/3", Qt.AlignBottom | Qt.AlignCenter, Qt.white)

    # 여기에 버전 체크 로직 결과에 따라 메시지 업데이트
    splash.showMessage("최신 버전 사용 중", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    time.sleep(2)



class ExceptionHandler:
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error("예외 발생", exc_info=(exc_type, exc_value, exc_traceback))


# exe 파일을 만들었을때 경로 인식을 위한 함수
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



if __name__ == '__main__':
    # 로그 설정
    logging.basicConfig(filename='gaussian_fit_error_log.txt', level=logging.ERROR, 
                        
                        format='%(asctime)s:%(levelname)s:%(message)s')
    sys.excepthook = ExceptionHandler().handle_exception
    app = QApplication(sys.argv)

    splash_pix = QPixmap(resource_path('assets/fingerPrint.png'))
    splash = QSplashScreen(splash_pix)
    splash.show()

    check_version(splash)

    main_win = DataFittngApp()

    splash.finish(main_win)
    sys.exit(app.exec_())


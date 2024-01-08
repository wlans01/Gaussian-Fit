import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit ,minimize_scalar


# 가우시안 함수 정의
def gaussian(x, amplitude, mean, stddev , y0):
    return y0 + amplitude * np.exp(-((x - mean) ** 2) / (2 * (stddev ** 2)))


def fitting(data ,data_name = "Line Profile",draw  = True):
    '''
    가우시안 피팅 함수

    data : pandas dataframe
    data_name : 그래프 이름
    draw : boolean  True : draw graph , False : not draw graph

    return :
    popt : fitting parameter  gaussian 함수의 parameter 3개 (amplitude, mean, stddev)
    fwhm : fwhm
    x : fitting data x
    gaussian(x, *popt) : fitting data y
    e2 : 1/e^2
    '''
    # pandas dataframe  numpy array로 변환
    data_numpy = data.to_numpy()

    # x,y 분리
    x = data_numpy[:,0]
    y = data_numpy[:,1]

    # 가우시안 피팅 수행
    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), np.mean(x), np.std(x)])

    # FWHM 계산
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]

    # 1/e^2  2w
    e2 = 1.699 * fwhm

    # 그래프 그리기
    if draw == True:

        plt.plot(data_numpy[:,0],data_numpy[:,1], label = data_name , color = 'black')
        plt.plot(data_numpy[:,0], gaussian(data_numpy[:,0], *popt), 'r--', label='Gaussian Fit')
        plt.title(f'{data_name}\nFWHM = {round(fwhm,3)}\n2w = {round(e2,3)}' ) 
        plt.grid(True)
        plt.legend()
        plt.show()

    fit_y = gaussian(x, *popt)

    return popt, fwhm ,x , fit_y , e2




def saturation_fitting(data,saturation_limit = -1 ,data_name = "Saturation", draw = True):
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
    


    # pandas dataframe  numpy array로 변환
    data_numpy = data.to_numpy()

    if saturation_limit == -1:
        is_saturation = False
    else : is_saturation = True


    ######## saturation 예측 피팅 #############3

    # saturation이 생기는 부분을 제거
    if not is_saturation:
        saturation_limit = max(data_numpy[:, 1])
    unsaturated_data = data_numpy[data_numpy[:, 1] <= saturation_limit]

    # saturation이 안생긴 데이터를 x와 y로 분리
    x = unsaturated_data[:, 0]
    y = unsaturated_data[:, 1]
    
    # 가우시안 피팅 수행
    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), np.mean(x), np.std(x),0])

    # 
    x_full = np.linspace(min(data_numpy[:, 0]), max(data_numpy[:, 0]), 5000)
    y_full = gaussian(x_full, *popt)

    # FWHM 계산
    # fwhm = abs(2 * np.sqrt(2 * np.log(2)) * popt[2])
    fwhm = abs(2 * np.sqrt( np.log(4)) * popt[2])

    # 1/e^2  2w
    e2 = 1.699 * fwhm


    ############## saturation 일반 피팅 ####################


    mean = popt[1]
    y0 = popt[3]
    print(y0)

    # FWHM의 시작 및 끝 좌표 계산
    fwhm_start_x = mean - fwhm / 2
    fwhm_end_x = mean + fwhm / 2
    fwhm_y = gaussian(mean, *popt) / 2  + y0

    # 1/e^2의 시작 및 끝 좌표 계산
    e2_start_x = mean - e2 / 2
    e2_end_x = mean + e2 / 2
    e2_y = gaussian(mean, *popt) / np.exp(2) + y0

    
    if draw == True:
        # saturation을 예측한 피팅
        plt.plot(data_numpy[:,0],data_numpy[:,1] , label = data_name , color = 'black')

        if is_saturation:
            plt.axhline(saturation_limit , color = 'gray' , linestyle = '--' , label = 'saturation limit')

        plt.plot([fwhm_start_x, fwhm_end_x], [fwhm_y, fwhm_y], 'g--', label='FWHM')
        plt.plot([e2_start_x, e2_end_x], [e2_y, e2_y], 'b--', label='1/e^2 (2w)')

        plt.plot(x_full,y_full, 'r--', label='Gaussian Fit')
        plt.title(f'{data_name}\nFWHM = {round(fwhm,3)}\n1/e^2 = {round(e2,3)}')
        plt.xlabel('Distance(pixel)')
        plt.ylabel('GrayValue')
        plt.grid(True)
        plt.legend()
        plt.show()



    return popt, fwhm , x_full , y_full ,x , y ,e2





if __name__ == '__main__':
    # # Example code
    # # read data
    # data1 = pd.read_excel(r'C:\Users\wlans\Desktop\fitting\Line Profile.xlsx')
    # data2 = pd.read_excel('saturation.xlsx')


    # # fitting
    # popt, fwhm ,x , fit_y , e2 = fitting(data1,data_name = "Line Profile",draw  = True)
    # popt, fwhm , x_full , y_full ,x , y ,e2 = saturation_fitting(data2,250,data_name = "Saturation",draw  = True)
   
    
    
    data1 = pd.read_excel(r'C:\Users\wlans\Desktop\fitting\Line Profile.xlsx')
    data2 = pd.read_excel('saturation.xlsx')
    data3 = pd.read_excel('testdata.xlsx')
    popt, fwhm , x_full , y_full ,x , y ,e2 = saturation_fitting(data1,data_name = "Saturation",draw  = True)
    popt, fwhm , x_full , y_full ,x , y ,e2 = saturation_fitting(data1,data_name = "Saturation",draw  = True)
   


    
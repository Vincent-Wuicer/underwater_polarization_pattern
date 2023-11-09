import numpy as np
import math
import cv2  # opencv支持与计算机视觉和机器学习相关的众多算法
from ziwuxian import Polarized_process #fitAzimuth
from scipy import stats  # 统计和随机数： scipy.stats;统计数据；配合numpy使用；模块scipy.stats包含随机过程的统计工具和概率描述
from shiyaying import syy
import os  # os模块包含普遍的操作系统功能,os.getcwd() 获取当前工作路径,os.listdir 以列表的形式展现当前目录或指定目录下的文件
from mask_add_photo import ronghe
import torch
from test1 import test_fun

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)    
else:
    device = torch.device("cpu")
print("Using {} device".format(device))
#torch.cuda.device_count()

def yu_nh(Aop_last,ang_jingdu,yuzhi,name,bo):   #提出创新性的ASPP拟合方法  
    pi = math.pi
    Aop_last_nom = np.uint8((Aop_last + pi / 2) / pi * 255)
    Aop_last_color = cv2.applyColorMap(Aop_last_nom, cv2.COLORMAP_JET) #伪色彩函数：applyColorMap()，转换为色度图
    if bo == 1:
        cv2.imwrite('data/pianzhen/pianzhen_'+name, Aop_last_color) #1代表分割后的
    else:
        cv2.imwrite('data/shiyaying/syy_'+name, Aop_last_color) #0代表分割后增强的
    num_x = Aop_last.shape[1] / 2 + 0.00001 #列
    num_y = Aop_last.shape[0] / 2 #行
    roi = num_x*4/4
    z_j = f_j = np.array([])
    z_l = f_l = np.array([])
    pi = math.pi
    zheng_shai = (Aop_last>yuzhi)
    fu_shai = Aop_last<-yuzhi
    huanshu = 10
    for aop_x in range(int(num_x - roi), int(num_x + roi)):
        for aop_y in range(int(num_y - roi), int(num_y + roi)):
            if (aop_x - num_x) ** 2 + (aop_y - num_y) ** 2 < roi ** 2:
                if zheng_shai[aop_y][aop_x]:
                    j_ang = math.atan((aop_y-num_y)/(aop_x-num_x))
                    z_juli = math.sqrt((aop_y-num_y)**2+(aop_x-num_x)**2)
                    z_j = np.append(z_j,j_ang)
                    z_l = np.append(z_l, z_juli)
                if fu_shai[aop_y][aop_x]:
                    f_ang = math.atan((aop_y-num_y)/(aop_x-num_x))
                    f_juli = math.sqrt((aop_y - num_y) ** 2 + (aop_x - num_x) ** 2)
                    f_j = np.append(f_j,f_ang)
                    f_l = np.append(f_l, f_juli)
    zf_l = np.append(z_l,f_l)
    zf_l.sort()
    kuan = zf_l[0:-1:int(len(zf_l)/huanshu)] #consequence[start_index: end_index: step]
    print(kuan)
    yh_zj = yh_fj = np.array([])
    shu_z = shu_f = np.array([0])
    qa = np.digitize(z_l, kuan) #numpy.digitize(x, bins, right = False)，返回x在bins中的位置；bins:一个一维的数组，必须是升序或者降序
    qa1 = np.digitize(f_l, kuan)
    for i in range(1,len(kuan)+1):
        sublist = z_j[np.where(qa == i)]
        sublist1 = f_j[np.where(qa1 == i)]#np.where(condition) 当where内条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
        yh_zj = np.append(yh_zj, sublist)
        yh_fj = np.append(yh_fj, sublist1)
        shu_z = np.append(shu_z, len(sublist)+shu_z[-1])
        shu_f = np.append(shu_f, len(sublist1)+shu_f[-1])
    if len(z_j) != 0 and len(f_j) != 0 :
        min_ji1 = 1000000
        zj1 = (z_j*100).astype(int)
        most_z = stats.mode(zj1)[0][0]/pi*1.8 #返回传入数组/矩阵中最常出现的成员以及出现的次数。[]表示维数
        fj1 = (f_j * 100).astype(int)
        most_f = stats.mode(fj1)[0][0]/pi*1.8# 计算正负特征点角度众数，子午线在众数附近旋转 # 返回众数
        for ang in np.arange(min(most_f,most_z)-20 ,max(most_f,most_z)+20, ang_jingdu): 
            kz = ang / 180 * pi
            kz1 = kz - 45 / 180 * pi
            kz2 = kz + 45 / 180 * pi
            i0 =shu_z[0]
            j0 = shu_f[0]
            jishu = 0
            zheng_xuan1 = ((z_j<kz) & (z_j > kz1)).sum() #&会将左右两个整数转换为二进制进行计算，当同位都为1时取1，否则取0
            fu_xuan1 = ((kz < f_j) & (f_j < kz2)).sum()
            zong_num1 = zheng_xuan1 + fu_xuan1
            for i,j in zip(shu_z[1:],shu_f[1:]): #for循环有两个需要迭代的对象时，要用zip对这多个变量封装;#[1:]意思是去掉列表中第一个元素（下标为0），去后面的元素进行操作
                if i != 0 :
                    zheng_xuan11 = ((kz > yh_zj[i0:i-1]) & (yh_zj[i0:i-1] > kz1)).sum()
                else:
                    zheng_xuan11 = 0
                if j != 0 :
                    fu_xuan11 = ((kz < f_j[j0:j-1]) & (f_j[j0:j-1] < kz2)).sum()
                else:
                    fu_xuan11 = 0
                zong_num11 = zheng_xuan11+fu_xuan11
                if zong_num11 != 0:
                    w = 1-abs(zheng_xuan11 / zong_num11-fu_xuan11 / zong_num11)*zong_num11/zong_num1
                    jishu1 = (zheng_xuan11 / zong_num11 * len(f_j[j0:j-1]) - fu_xuan11 / zong_num11 * len(yh_zj[i0:i-1]))*w
                else:
                    jishu1 = 0
                i0 = i
                j0 = j
                jishu = jishu+jishu1
            if abs(jishu) < abs(min_ji1):
                min_ji1 = jishu
                best_k1 = math.tan(kz)  #拟合子午线的斜率
                best_ang1 = ang
        Aop_last_color = cv2.line(Aop_last_color, (10, int((10 - num_x) * best_k1 + num_y)), (1014, int((1014 - num_x) * best_k1 + num_y)), (0, 255, 0), 4)
        # cv2.line(image, start_point, end_point, color, thickness) (0,255,0)代表绿色的颜色，4代表线的粗细；
        for i in kuan:
            Aop_last_color = cv2.circle(Aop_last_color, (int(num_x), int(num_y)), int(i), (0, 0, 0), 2)
        # Aop_last_color = cv2.circle(Aop_last_color, (int(num_x),int(num_y)),int(roi), (255, 255, 255), 4) 
        # 圆绘制：cv2.circle(img,center,radius,color,thickness=None,lineType=None)；center表示圆心，radius表示半径
        if bo == 1:
            cv2.imwrite('data/result/chushi/'+name, Aop_last_color)
        else:
            cv2.imwrite('data/result/zengqiang/'+name, Aop_last_color)
        return best_k1,best_ang1
    else:
        return 0,0

if __name__ == "__main__":
    # img1 = cv2.imread('data/chushi/368.jpg')
    img_path = 'data/page/3/tree/chushi/' 
    test_path = 'data/page/3/tree/xunlian/'
    save_path = 'data/page/3/tree/ronghe/'
    model_path = "model/xmodel_50_50.pth"  # 保存的模型文件的路径
    test_fun(img_path,model_path,test_path) #图像训练  U-net分割
    ronghe(chushi_dir=img_path,mask_dir=test_path,savepath=save_path)
    img_path1 = os.listdir(save_path)
    for i in img_path1:
        img1 = cv2.imread(save_path+i) #读取到保存的融合图片，即Unet分割后的目标天空区域图像
        if img1.shape[1] > 2048:  #裁剪2048以外的部分;x.shape[0]输出行数；x.shape[1] 输出列数 
            img1 = img1[:, 200:2248, 0]
        else:
            img1 = img1[:, :, 0]            
        aop0, _ = Polarized_process(img1) #Unet分割后的偏振图            
        img2 = syy(img1) #原图增强
        aop1, _ = Polarized_process(img2)   #原图增强后的偏振图   
        # aop1 = syy(aop0)
        ang_jingdu = 0.3
        yuzhi = 87 / 180 * math.pi     
        best_k1, best_ang1 = yu_nh(aop0, ang_jingdu, yuzhi,i, 1)   
        best_k, best_ang = yu_nh(aop1, ang_jingdu, yuzhi - 5 / 180 * math.pi,i, 0)       
        # ransac_k,old_k = fitAzimuth(aop1)
        print('yuan_rst: ', best_ang1)
        print('youhua_rst: ', best_ang)
        with open('data/result/data.txt', 'w') as f:  #写w, 阅读R；a: 追加内容，用write() 会在已经写的内容基础上追加新的内容
            f.write('%s yuan_rst:%f youhua_rst:%f'%(i,best_ang1,best_ang))   #打开txt文件，写入txt文件
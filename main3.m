%% 清空环境变量
clc;
clear;
close all;
% 	// Output image:
% 				//   -------------
% 				//   |  90 |  45 |
% 				//   |-----------|
% 				//   | 135 |   0 |
% 				//   -------------
%% 读图
% img_0 = imread('1.jpg');
% img_45 = imread('3.jpg');
% img_90 = imread('4.jpg');
% img_135 = imread('2.jpg'); 
img = imread('1.jpg');
img=rgb2gray(img);
img_90 = img(1:2:end,1:2:end);
img_135 = img(2:2:end,1:2:end);
img_45 = img(1:2:end,2:2:end);
img_0 = img(2:2:end,2:2:end);
figure;imshow(img);
%% 偏振信息获取AOP，DOP
I1=double(img_0);
I2=double(img_45);
I3=double(img_90);
I4=double(img_135); 

I = max((I1+I3),(I2+I4));
Q=(I1-I3);
U=(2*I2-I1-I3);
DOP = sqrt((Q.^2+U.^2))./I;DOP=double(DOP);
AOP=0.5*atan2(U,Q);AOP=double(AOP);% 偏振角度
PI = 3.1415926;
AOP_degree = AOP * 180/PI;
% imwrite(AOP, 'test.jpg');
figure;imagesc(AOP);
figure;imagesc(DOP);
%% 画圆
% 图像分辨率：1024*1224
% 天顶点（由RSG+形态学操作确定）（i0，j0）：（275 ，585）
P_y_mean = 512;
P_x_mean = 612;
rr = 200; % 半径-
for i = 1:1024
    for j = 1:1224
        if((i - P_y_mean)*(i - P_y_mean) + (j - P_x_mean)*(j - P_x_mean)<rr*rr)
            AOP_circle(i,j) = AOP(i,j);
        else
            AOP_circle(i,j) = NaN;
        end
    end
end

for i = 1:1024
    for j = 1:1224
        if((i - P_y_mean)*(i - P_y_mean) + (j - P_x_mean)*(j - P_x_mean)<rr*rr)
            DOP_circle(i,j) = DOP(i,j);
        else
            DOP_circle(i,j) = NaN;
        end
    end
end
%% 显示
r = 399;
c = 399;
rect=[P_x_mean-rr P_y_mean-rr r c]; %(x,y)以及长度（长度可调节）
AOP_cut=imcrop(AOP_circle,rect); %指定裁剪区域裁剪图像
DOP_cut=imcrop(DOP_circle,rect); %图像裁剪
figure;imagesc(AOP_circle);
figure;imagesc(AOP_cut);
figure;imagesc(DOP_circle);
figure;imagesc(DOP_cut);
%% 重绘制偏振角图,使其沿着太阳子午线中心反对称且呈∞字形分布
for i =1:1024
    for j = 1:1224
        AOP_last(i,j) = AOP_circle(i,j) - atan((i - 512)/(612 - j)); 
        if AOP_last(i,j) < -PI/2
            AOP_last(i,j) = AOP_last(i,j) + PI;
        elseif AOP_last(i,j) >PI/2
            AOP_last(i,j) = AOP_last(i,j) - PI;
        end
    end
end
% AOP_last=flipud(AOP_last);%把矩阵左右翻转
AOP_cut=imcrop(AOP_last,rect);
figure;imagesc(AOP_cut);
%% 随机取点计算E矢量
% 随机点数目：1000
% a1(1)-----列；a1(2)-----行
% 焦距：fc
amount = 10000;
num_efficient = 0;
fc = 3.5;% 焦距3.5mm
 for i = 1:amount
    % 生成随机矩阵
    a1 = ceil(rand(1,2)*600); 
    a2 = ceil(rand(1,2)*800);
    % 判断是否在圆内
    if ((a1(1) - 512)*(a1(1) - 512) + (a1(2) - 612)*(a1(2) - 612)<rr*rr)&&...
       ((a2(1) - 512)*(a2(1) - 512) + (a2(2) - 612)*(a2(2) - 612)<rr*rr)
        num_efficient = num_efficient+1;
        % E矢量坐标表示
%         a1(1) = 400;
%         a1(2) = 612;
        AOP1 = AOP_last(a1(1),a1(2)); % 随机点偏振角度值
        AOP2 = AOP_last(a2(1),a2(2));
        E_vector1 = [cos(AOP1),sin(AOP1),0]; % 入射光坐标系E矢量坐标表示
        E_vector2 = [cos(AOP2),sin(AOP2),0];
        % 坐标变换矩阵
        cols1_distance = (a1(1) - 512)*2.4800000000000e-02;% y1-yc
        cols2_distance = (a2(1) - 512)*2.4800000000000e-02;% y2-yc
        rows1_distance = (a1(2) - 612)*2.0800000000000e-02;% x1-xc
        rows2_distance = (a2(2) - 612)*2.0800000000000e-02;
        
        a_angle1 = atan(sqrt(cols1_distance*cols1_distance+rows1_distance*rows1_distance)/fc);
        a_angle2 = atan(sqrt(cols2_distance*cols2_distance+rows2_distance*rows2_distance)/fc);
        a_angle11(num_efficient) = a_angle1*180/3.14;
        a_angle22(num_efficient) = a_angle2*180/3.14;
        b_angle1 = atan(cols1_distance/(rows1_distance+0.0001));
        b_angle2 = atan(cols2_distance/(rows2_distance+0.0001));
        b_angle11(num_efficient) = b_angle1*180/3.14;
        b_angle22(num_efficient) = b_angle2*180/3.14;
        C11 = [cos(a_angle1),0,sin(a_angle1);0,1,0;-sin(a_angle1),0,cos(a_angle1)];
        C12 = [cos(b_angle1),sin(b_angle1),0;-sin(b_angle1),cos(b_angle1),0;0,0,1];
        C1 = C12*C11;
        
        C21 = [cos(a_angle2),0,sin(a_angle2);0,1,0;-sin(a_angle2),0,cos(a_angle2)];
        C22 = [cos(b_angle2),sin(b_angle2),0;-sin(b_angle2),cos(b_angle2),0;0,0,1];
        C2 = C22*C21;
        
        % 坐标变换
        e1_last = C1* E_vector1';
        e2_last = C2* E_vector2';
        e1_last(3)  =  e1_last(3)+ sin(b_angle1);
        e2_last(3)  =  e2_last(3)+ sin(b_angle2);
%         e1_last = E_vector1;
%         e2_last = E_vector2;
        % 太阳矢量(E矢量叉乘)
        sun_vector(num_efficient,:) = (cross(e1_last,e2_last));% 此 MATLAB 函数 返回 A 和 B 的叉积。
        sun_angle(num_efficient) = atan2(sun_vector(num_efficient,1),sun_vector(num_efficient,2))*180/PI;
%         if (sun_angle(num_efficient)<0)
%             sun_angle(num_efficient) = sun_angle(num_efficient)+180;
%         end
  
        sun_H(num_efficient) = atan2(sun_vector(num_efficient,3),...
        sqrt((sun_vector(num_efficient,1))^2 + (sun_vector(num_efficient,2))^2))*180/PI;
%         if (sun_H(num_efficient)<0)
%             sun_H(num_efficient) = abs(sun_H(num_efficient));
%         end
    end
 end
 figure;plot(sun_H);
figure;plot(sun_angle);
 %% 拟合方位角和高度角
%         最小二乘法
% x = 1:num_efficient;
% y = sun_angle;
% figure;plot(sun_angle);
% coefficient = polyfit(x,y,1);
% y1 = polyval(coefficient,x);
% hold on;plot(x,y1,'r-','LineWidth',1);
% 
% % RANSAC直线拟合
% points(:,1) = x;
% points(:,2) = y;
% sampleSize = 2; % number of points to sample per trial
% maxDistance = 2; % max allowable distance for inliers
% 
% fitLineFcn = @(points) polyfit(points(:,1),points(:,2),1); % fit function using polyfit
% evalLineFcn = ...   % distance evaluation function
%   @(model, points) sum((points(:, 2) - polyval(model, points(:,1))).^2,2);
% 
% [modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
%   sampleSize,maxDistance);
% 
% modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
% inlierPts = points(inlierIdx,:);
% x = [min(inlierPts(:,1)) max(inlierPts(:,1))];
% y = modelInliers(1)*x + modelInliers(2);
% hold on;plot(x, y, 'g-','LineWidth',2)
% title(['bestLine:  y =  ',num2str(modelInliers(1)),'x + ',num2str(modelInliers(2))]);

ir_img=imread('Kaptein_1123_IR.bmp');
vis_img=imread('Kaptein_1123_Vis.bmp');

% 转换为灰度图（如果是彩色图像）
if size(ir_img, 3) == 3
    ir_img = rgb2gray(ir_img)
end
if size(vis_img, 3) == 3
    vis_img = rgb2gray(vis_img);
end

% 分别计算高度和宽度
h = min(size(ir_img, 1), size(vis_img, 1));
w = min(size(ir_img, 2), size(vis_img, 2));

% 统一尺寸
ir_img1= imresize(ir_img, [h, w]);
vis_img1= imresize(vis_img, [h, w]);

imshow(ir_img1);figure;imshow(vis_img1);


%去噪
h_gauss=fspecial('gaussian',[3,3],0.5);
ir_img=imfilter(ir_img1,h_gauss);
figure;subplot(2,2,1);imshow(ir_img1);title('滤波前的红外图像');
subplot(2,2,2);imshow(ir_img);title('滤波后的红外图像');
vis_img=imfilter(vis_img1,h_gauss);
subplot(2,2,3);imshow(vis_img1);title('滤波前的可将光图像');
subplot(2,2,4);imshow(vis_img);title('滤波后的可见光图像');

%对比度增强（直方直方图均衡化)
img_vis_eq=histeq(vis_img);
figure;imshow(img_vis_eq);title('直方图均衡化后的图像');
vis_img=img_vis_eq;

%%边缘增强
% 可见光边缘增强（拉普拉斯锐化）
laplacian_kernel = [0 -1 0; -1 5 -1; 0 -1 0];
vis_sharpened = imfilter(vis_img, laplacian_kernel);
figure;imshow(vis_sharpened);title('边缘增强后的可见光图');
vis_img=vis_sharpened;


% 小波分解
[ir_cA, ir_cH, ir_cV, ir_cD] = dwt2(ir_img, 'haar');
[vis_cA, vis_cH, vis_cV, vis_cD] = dwt2(vis_img, 'haar');

% 二级分解（新增部分）
[ir_cA2, ir_cH2, ir_cV2, ir_cD2] = dwt2(ir_cA, 'haar');
[vis_cA2, vis_cH2, vis_cV2, vis_cD2] = dwt2(vis_cA, 'haar');

% 1. 二级低频融合（fused_cA2）
heat_mask2 = ir_cA2 > 0.5*max(ir_cA2(:)); % 二级热目标掩码
alpha = 0.85;
fused_cA2 = alpha*ir_cA2.*heat_mask2 + (1-alpha)*vis_cA2.*~heat_mask2 + ...
            0.5*(ir_cA2 + vis_cA2).*(~(heat_mask2 | ~heat_mask2));

% 低频取红外（突出热目标），高频取绝对值大者
%通过阈值生成热目标掩码，
% 根据红外热目标显著性动态调整权重
heat_mask = ir_cA > 0.5*max(ir_cA(:)); % 假设高温区域占前50%
alpha = 0.85; % 基础权重(红外图像占比)
fused_cA = alpha*ir_cA.*heat_mask + (1-alpha)*vis_cA.*~heat_mask + ...
           0.5*(ir_cA + vis_cA).*(heat_mask==0 & heat_mask==0); 
% 过渡区域
%通过阈值生成热目标掩码，在热目标区域主要保留红外信息，其他区域保留可见光信息，中间区域取平均值。
       
% 一级高频系数融合
fused_cH = (abs(ir_cH) > abs(vis_cH)) .* ir_cH + (abs(ir_cH) <= abs(vis_cH)) .* vis_cH;
fused_cV = (abs(ir_cV) > abs(vis_cV)) .* ir_cV + (abs(ir_cV) <= abs(vis_cV)) .* vis_cV;
fused_cD = (abs(ir_cD) > abs(vis_cD)) .* ir_cD + (abs(ir_cD) <= abs(vis_cD)) .* vis_cD;

% 高频系数融合：比较红外和可见光图像的高频系数绝对值，选择更大的一方（即保留更明显的边缘和纹理）。
fused_cH2 = (abs(ir_cH2) > abs(vis_cH2)) .* ir_cH2 + (abs(ir_cH2) <= abs(vis_cH2)) .* vis_cH2;
fused_cV2 = (abs(ir_cV2) > abs(vis_cV2)) .* ir_cV2 + (abs(ir_cV2) <= abs(vis_cV2)) .* vis_cV2;
fused_cD2 = (abs(ir_cD2) > abs(vis_cD2)) .* ir_cD2 + (abs(ir_cD2) <= abs(vis_cD2)) .* vis_cD2;
% 重构图像
enhanced_cA1 = idwt2(fused_cA, fused_cH, fused_cV, fused_cD, 'haar');
% 二级重构（从二级分解结果恢复到一级低频系数）
enhanced_cA = idwt2(fused_cA2, fused_cH2, fused_cV2, fused_cD2, 'haar');

% 最终重构（结合一级高频系数生成完整图像）
fused_img = idwt2(fused_cA, fused_cH, fused_cV, fused_cD, 'haar');

imwrite(uint8(fused_img), 'fused_result.jpg');


figure;subplot(1,3,1);imshow(ir_img1);title('红外图像');
subplot(1,3,2);imshow(vis_img1);title('可见光图像');
subplot(1,3,3);imshow(uint8(fused_img));title('融合图像');




% 图像融合质量评估指标计算
% 读取融合图像
fusedImg = imread('fused_result.jpg');
if size(fusedImg, 3) == 3
    fusedImg = rgb2gray(fusedImg); % 转为灰度
end

% 计算信息熵
[counts, ~] = imhist(fusedImg);
prob = counts / sum(counts);
prob = prob(prob > 0); % 移除0值避免log(0)
EN = -sum(prob .* log2(prob));
disp(['信息熵 (EN): ', num2str(EN)]);

% 转为double类型计算梯度
img = double(fusedImg);
[m, n] = size(img);

% 计算行频率 (RF) 和列频率 (CF)
RF = sqrt(sum(sum((img(:, 2:n) - img(:, 1:n-1)).^2)) / (m*n));
CF = sqrt(sum(sum((img(2:m, :) - img(1:m-1, :)).^2)) / (m*n));
SF = sqrt(RF^2 + CF^2);
disp(['空间频率 (SF): ', num2str(SF)]);

% 计算x和y方向梯度
[Gx, Gy] = gradient(img);
AG = mean2(sqrt(Gx.^2 + Gy.^2));
disp(['平均梯度 (AG): ', num2str(AG)]);

SD = std2(fusedImg);
disp(['标准差 (SD): ', num2str(SD)]);



% 读取源图像
srcImg1 = ir_img1;
srcImg2 = vis_img;
if size(srcImg1, 3) == 3
    srcImg1 = rgb2gray(srcImg1);
    srcImg2 = rgb2gray(srcImg2);
end

% 计算联合直方图
jointHist1 = histcounts2(double(srcImg1), double(fusedImg), 256, 'Normalization', 'probability');
jointHist2 = histcounts2(double(srcImg2), double(fusedImg), 256, 'Normalization', 'probability');

% 计算互信息
marginal1 = sum(jointHist1, 2);
marginalF1 = sum(jointHist1, 1);
MI1 = jointHist1 .* log2(jointHist1 ./ (marginal1 .* marginalF1 + eps));
MI1 = nansum(MI1(:));

marginal2 = sum(jointHist2, 2);
marginalF2 = sum(jointHist2, 1);
MI2 = jointHist2 .* log2(jointHist2 ./ (marginal2 .* marginalF2 + eps));
MI2 = nansum(MI2(:));

MI = MI1 + MI2;
disp(['互信息 (MI): ', num2str(MI)]);
    
% 显示结果
disp('===== 融合图像评估结果 =====');
disp(['信息熵 (EN): ', num2str(EN)]);
disp(['空间频率 (SF): ', num2str(SF)]);
disp(['平均梯度 (AG): ', num2str(AG)]);
disp(['标准差 (SD): ', num2str(SD)]);
disp(['互信息 (MI): ', num2str(MI)]);





% 读取黑白图像
grayImage = imread('fused_result.jpg'); % 替换为你的图像文件名

% 确保图像是灰度图
if size(grayImage, 3) > 1
    grayImage = rgb2gray(grayImage);
end

% 创建图像窗口
figure('Position', [100, 100, 1200, 500]);

% 显示原始黑白图像
subplot(1, 1, 1);
imshow(grayImage);
title('融合后的伪彩色图像');

% 区域分割（使用Otsu阈值）
level = graythresh(fused_img);
binaryImage = imbinarize(fused_img, level);

% 创建自定义颜色映射（低温区蓝色，高温区红色）
tempMap = zeros(256, 3);
for i = 1:256
    if i < 128
        % 低温区域 - 蓝色渐变
        tempMap(i, :) = [0, 0, i/128];
    else
        % 高温区域 - 红色渐变
        tempMap(i, :) = [(i-128)/128, 0, 0];
    end
end

% 应用自适应伪彩色
%subplot(1, 2, 2);
%figure;
%imshow(fused_img);
colormap(tempMap);
colorbar;
%title('自适应温度伪彩色');






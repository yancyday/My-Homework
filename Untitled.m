ir_img=imread('Kaptein_1123_IR.bmp');
vis_img=imread('Kaptein_1123_Vis.bmp');

% ת��Ϊ�Ҷ�ͼ������ǲ�ɫͼ��
if size(ir_img, 3) == 3
    ir_img = rgb2gray(ir_img)
end
if size(vis_img, 3) == 3
    vis_img = rgb2gray(vis_img);
end

% �ֱ����߶ȺͿ��
h = min(size(ir_img, 1), size(vis_img, 1));
w = min(size(ir_img, 2), size(vis_img, 2));

% ͳһ�ߴ�
ir_img1= imresize(ir_img, [h, w]);
vis_img1= imresize(vis_img, [h, w]);

imshow(ir_img1);figure;imshow(vis_img1);


%ȥ��
h_gauss=fspecial('gaussian',[3,3],0.5);
ir_img=imfilter(ir_img1,h_gauss);
figure;subplot(2,2,1);imshow(ir_img1);title('�˲�ǰ�ĺ���ͼ��');
subplot(2,2,2);imshow(ir_img);title('�˲���ĺ���ͼ��');
vis_img=imfilter(vis_img1,h_gauss);
subplot(2,2,3);imshow(vis_img1);title('�˲�ǰ�Ŀɽ���ͼ��');
subplot(2,2,4);imshow(vis_img);title('�˲���Ŀɼ���ͼ��');

%�Աȶ���ǿ��ֱ��ֱ��ͼ���⻯)
img_vis_eq=histeq(vis_img);
figure;imshow(img_vis_eq);title('ֱ��ͼ���⻯���ͼ��');
vis_img=img_vis_eq;

%%��Ե��ǿ
% �ɼ����Ե��ǿ��������˹�񻯣�
laplacian_kernel = [0 -1 0; -1 5 -1; 0 -1 0];
vis_sharpened = imfilter(vis_img, laplacian_kernel);
figure;imshow(vis_sharpened);title('��Ե��ǿ��Ŀɼ���ͼ');
vis_img=vis_sharpened;


% С���ֽ�
[ir_cA, ir_cH, ir_cV, ir_cD] = dwt2(ir_img, 'haar');
[vis_cA, vis_cH, vis_cV, vis_cD] = dwt2(vis_img, 'haar');

% �����ֽ⣨�������֣�
[ir_cA2, ir_cH2, ir_cV2, ir_cD2] = dwt2(ir_cA, 'haar');
[vis_cA2, vis_cH2, vis_cV2, vis_cD2] = dwt2(vis_cA, 'haar');

% 1. ������Ƶ�ںϣ�fused_cA2��
heat_mask2 = ir_cA2 > 0.5*max(ir_cA2(:)); % ������Ŀ������
alpha = 0.85;
fused_cA2 = alpha*ir_cA2.*heat_mask2 + (1-alpha)*vis_cA2.*~heat_mask2 + ...
            0.5*(ir_cA2 + vis_cA2).*(~(heat_mask2 | ~heat_mask2));

% ��Ƶȡ���⣨ͻ����Ŀ�꣩����Ƶȡ����ֵ����
%ͨ����ֵ������Ŀ�����룬
% ���ݺ�����Ŀ�������Զ�̬����Ȩ��
heat_mask = ir_cA > 0.5*max(ir_cA(:)); % �����������ռǰ50%
alpha = 0.85; % ����Ȩ��(����ͼ��ռ��)
fused_cA = alpha*ir_cA.*heat_mask + (1-alpha)*vis_cA.*~heat_mask + ...
           0.5*(ir_cA + vis_cA).*(heat_mask==0 & heat_mask==0); 
% ��������
%ͨ����ֵ������Ŀ�����룬����Ŀ��������Ҫ����������Ϣ�������������ɼ�����Ϣ���м�����ȡƽ��ֵ��
       
% һ����Ƶϵ���ں�
fused_cH = (abs(ir_cH) > abs(vis_cH)) .* ir_cH + (abs(ir_cH) <= abs(vis_cH)) .* vis_cH;
fused_cV = (abs(ir_cV) > abs(vis_cV)) .* ir_cV + (abs(ir_cV) <= abs(vis_cV)) .* vis_cV;
fused_cD = (abs(ir_cD) > abs(vis_cD)) .* ir_cD + (abs(ir_cD) <= abs(vis_cD)) .* vis_cD;

% ��Ƶϵ���ںϣ��ȽϺ���Ϳɼ���ͼ��ĸ�Ƶϵ������ֵ��ѡ������һ���������������Եı�Ե��������
fused_cH2 = (abs(ir_cH2) > abs(vis_cH2)) .* ir_cH2 + (abs(ir_cH2) <= abs(vis_cH2)) .* vis_cH2;
fused_cV2 = (abs(ir_cV2) > abs(vis_cV2)) .* ir_cV2 + (abs(ir_cV2) <= abs(vis_cV2)) .* vis_cV2;
fused_cD2 = (abs(ir_cD2) > abs(vis_cD2)) .* ir_cD2 + (abs(ir_cD2) <= abs(vis_cD2)) .* vis_cD2;
% �ع�ͼ��
enhanced_cA1 = idwt2(fused_cA, fused_cH, fused_cV, fused_cD, 'haar');
% �����ع����Ӷ����ֽ����ָ���һ����Ƶϵ����
enhanced_cA = idwt2(fused_cA2, fused_cH2, fused_cV2, fused_cD2, 'haar');

% �����ع������һ����Ƶϵ����������ͼ��
fused_img = idwt2(fused_cA, fused_cH, fused_cV, fused_cD, 'haar');

imwrite(uint8(fused_img), 'fused_result.jpg');


figure;subplot(1,3,1);imshow(ir_img1);title('����ͼ��');
subplot(1,3,2);imshow(vis_img1);title('�ɼ���ͼ��');
subplot(1,3,3);imshow(uint8(fused_img));title('�ں�ͼ��');




% ͼ���ں���������ָ�����
% ��ȡ�ں�ͼ��
fusedImg = imread('fused_result.jpg');
if size(fusedImg, 3) == 3
    fusedImg = rgb2gray(fusedImg); % תΪ�Ҷ�
end

% ������Ϣ��
[counts, ~] = imhist(fusedImg);
prob = counts / sum(counts);
prob = prob(prob > 0); % �Ƴ�0ֵ����log(0)
EN = -sum(prob .* log2(prob));
disp(['��Ϣ�� (EN): ', num2str(EN)]);

% תΪdouble���ͼ����ݶ�
img = double(fusedImg);
[m, n] = size(img);

% ������Ƶ�� (RF) ����Ƶ�� (CF)
RF = sqrt(sum(sum((img(:, 2:n) - img(:, 1:n-1)).^2)) / (m*n));
CF = sqrt(sum(sum((img(2:m, :) - img(1:m-1, :)).^2)) / (m*n));
SF = sqrt(RF^2 + CF^2);
disp(['�ռ�Ƶ�� (SF): ', num2str(SF)]);

% ����x��y�����ݶ�
[Gx, Gy] = gradient(img);
AG = mean2(sqrt(Gx.^2 + Gy.^2));
disp(['ƽ���ݶ� (AG): ', num2str(AG)]);

SD = std2(fusedImg);
disp(['��׼�� (SD): ', num2str(SD)]);



% ��ȡԴͼ��
srcImg1 = ir_img1;
srcImg2 = vis_img;
if size(srcImg1, 3) == 3
    srcImg1 = rgb2gray(srcImg1);
    srcImg2 = rgb2gray(srcImg2);
end

% ��������ֱ��ͼ
jointHist1 = histcounts2(double(srcImg1), double(fusedImg), 256, 'Normalization', 'probability');
jointHist2 = histcounts2(double(srcImg2), double(fusedImg), 256, 'Normalization', 'probability');

% ���㻥��Ϣ
marginal1 = sum(jointHist1, 2);
marginalF1 = sum(jointHist1, 1);
MI1 = jointHist1 .* log2(jointHist1 ./ (marginal1 .* marginalF1 + eps));
MI1 = nansum(MI1(:));

marginal2 = sum(jointHist2, 2);
marginalF2 = sum(jointHist2, 1);
MI2 = jointHist2 .* log2(jointHist2 ./ (marginal2 .* marginalF2 + eps));
MI2 = nansum(MI2(:));

MI = MI1 + MI2;
disp(['����Ϣ (MI): ', num2str(MI)]);
    
% ��ʾ���
disp('===== �ں�ͼ��������� =====');
disp(['��Ϣ�� (EN): ', num2str(EN)]);
disp(['�ռ�Ƶ�� (SF): ', num2str(SF)]);
disp(['ƽ���ݶ� (AG): ', num2str(AG)]);
disp(['��׼�� (SD): ', num2str(SD)]);
disp(['����Ϣ (MI): ', num2str(MI)]);





% ��ȡ�ڰ�ͼ��
grayImage = imread('fused_result.jpg'); % �滻Ϊ���ͼ���ļ���

% ȷ��ͼ���ǻҶ�ͼ
if size(grayImage, 3) > 1
    grayImage = rgb2gray(grayImage);
end

% ����ͼ�񴰿�
figure('Position', [100, 100, 1200, 500]);

% ��ʾԭʼ�ڰ�ͼ��
subplot(1, 1, 1);
imshow(grayImage);
title('�ںϺ��α��ɫͼ��');

% ����ָʹ��Otsu��ֵ��
level = graythresh(fused_img);
binaryImage = imbinarize(fused_img, level);

% �����Զ�����ɫӳ�䣨��������ɫ����������ɫ��
tempMap = zeros(256, 3);
for i = 1:256
    if i < 128
        % �������� - ��ɫ����
        tempMap(i, :) = [0, 0, i/128];
    else
        % �������� - ��ɫ����
        tempMap(i, :) = [(i-128)/128, 0, 0];
    end
end

% Ӧ������Ӧα��ɫ
%subplot(1, 2, 2);
%figure;
%imshow(fused_img);
colormap(tempMap);
colorbar;
%title('����Ӧ�¶�α��ɫ');






% Display maize point clouds
pc=pcread('example_maize_pc.ply');
pcshow(pc)

% Meshing 
cmd=['E:\AQ\PCSR.exe "E:\AQ" "example_maize_pc.ply" "E:\AQ" "example_maize_pm.ply"']; 
system(cmd)
[group_sp,group_se]=read_44.22ply('example_maize_pm.ply');

% Display maize point clouds
pc=pcread('output.ply');
pcshow(pc)

%% Meshing 
%cmd=['E:\AQ\PCSR-git.exe "E:\AQ" "output.ply" "E:\AQ" "output1.ply"']; 
%system(cmd)
%[group_sp,group_se]=read_ply('A619-1-1-7.29-mesh.ply');
%[group_sp,group_se,label]=read_ply('A619-1-1-7.29-label.ply');
[group_sp,group_se,label]=read_ply('A619-1-1-7.29-f-label.ply');
%[group_sp,group_se,label]=read_ply('E:\AQ\DATA\mesh-label\M55-2-1-7.29-mesh-meshlabel.ply');

%disp('label:');
%disp(label);

%% Create virtual canopy 
row_num=4; plant_num=4;
plant_dis=0.2; row_dis=0.5;
[group_rp,group_re]=plant2canopy(group_se,group_sp,label,row_num,plant_num,plant_dis,row_dis);
figure
set(gca,'FontName','Times New Roman','FontSize',20);

trisurf(group_re,group_rp(:,1),group_rp(:,2),group_rp(:,3),'Facecolor',[0.133,0.545,0.133],'FaceAlpha', 0.5,'EdgeColor','none')
axis on
axis equal

canopy_facet_num=length(group_re);
input_mat=[ones(canopy_facet_num,1),group_rp(group_re(:,1),end),ones(canopy_facet_num,1),zeros(canopy_facet_num,1),...
    zeros(canopy_facet_num,1),group_rp(group_re(:,1),1:3).*100,group_rp(group_re(:,2),1:3).*100,group_rp(group_re(:,3),1:3).*100,...
    zeros(canopy_facet_num,1),ones(canopy_facet_num,1).*0.05,ones(canopy_facet_num,1).*0.05];
writematrix(input_mat,'canopy_model.txt','Delimiter',' ');

%%  Ray tracing 
cmd=['E:\AQ\fastTracerV1.22.exe -D 30 70 75 175 0 300 -L 31 -S 12 -A 0.7 -d 120 -W 13 1 13 -n 0.1 -m ','E:\AQ\canopy_model.txt',' -o ','E:\AQ\result.txt',' -z 0.5']; 
system(cmd);

%% A-Q curve fiting
A=[31.49 30.83 29.15 27.48 25.30 21.86 16.25 12.64 8.68 6.52 4.20 1.56 -2.81];%光合速率单位umol/m^2 s
PPFD=[2000 1600 1200 1000 800 600 400 300 200 150 100 50 0];%光强单位umol/m^2 s

para0=[0.1 30 1 0.5];  % initial parameters
[para_fit,resnorm]= lsqcurvefit(@A_Q_curve,para0,PPFD,A,[0 0 0 0], [+inf +inf +inf 1]);
PPFD_m=0:1:2500;
for i=1:length(PPFD_m)
A_m(i) =A_Q_curve(para_fit,PPFD_m(i));
end

figure
set(gca,'FontName','Times New Roman','FontSize',20);
plot(PPFD,A,'*')
hold on
plot (PPFD_m, A_m,'-')

%% canopy light interception and photosynthetic rate
result=readmatrix('E:\AQ\result.txt');
pos=result(:,6:14);
pos_all=[pos(:,1:3);pos(:,4:6);pos(:,7:9)];
%提取标签列 第二列
labels = result(:, 2);
faces=[(1:length(pos))',(length(pos)+1:length(pos)*2)',(length(pos)*2+1:length(pos)*3)'];
disp('size(faces)')
disp(size(faces))
%面积单位为平方米
facet_area=result(:,18)/10000;
PPFD_c=result(:,25);


figure
set(gca,'FontName','Times New Roman','FontSize',20);
trisurf(faces,pos_all(:,1),pos_all(:,2),pos_all(:,3),PPFD_c,'FaceAlpha',0.9,'EdgeColor','none'),caxis([0 2000])
set(gcf,'Color',[1,1,1])
axis equal
colorbar

A_mc=zeros(size(PPFD_c))

for i=1:length(PPFD_c)
A_mc(i) =A_Q_curve(para_fit,PPFD_c(i,1));
end



figure
set(gca,'FontName','Times New Roman','FontSize',20);
trisurf(faces,pos_all(:,1),pos_all(:,2),pos_all(:,3),A_mc,'FaceAlpha', 0.9,'EdgeColor','none')
set(gcf,'Color',[1,1,1])
axis equal
colorbar
disp('PPFD-C');

groud_area=(70-30)*(175-75)/10000;
A_c=sum(A_mc.*facet_area)/groud_area ; % Total canopy photosynthetic rate 单位umol/m^2 s 
%disp(['Total Canopy Photosynthetic Rate: ' num2str(A_c)]);
disp(['总光合速率 ' num2str(A_c)]);
%disp(size(A_c));
% 计算每单位地面日光截获量 LI
LI = sum(PPFD_c .* facet_area) / groud_area; % 单位 umol/m^2 s
disp(['每单位地面日光截获量 LI: ' num2str(LI)]);
% 计算光利用效率 LUE
LUE = A_c / LI;
disp(['光利用效率 LUE: ' num2str(LUE)]);
% 计算每个面片的光合有效光子通量密度（PPFD_c）与面积的乘积，即吸收的光量
light_absorbed = PPFD_c .* facet_area;
disp(['总吸收光量: ' num2str(sum(light_absorbed))]);


% 获取唯一的标签值
unique_labels = unique(labels);

% 存储每个类别的 A_c
A_c_per_class = zeros(size(unique_labels));

% 存储每个类别的吸收光量
light_absorbed_per_class = zeros(size(unique_labels));

% 存储每个类别的光能利用效率
LUE_per_class = zeros(size(unique_labels));

% 计算每个类别的 A_c
for i = 1:length(unique_labels)
    % 找到属于当前类别的索引
    indices = find(labels == unique_labels(i));
    
    % 提取当前类别的 A_mc 和 facet_area
    A_mc_class = A_mc(indices);
    facet_area_class = facet_area(indices);
    
    % 计算当前类别的 A_c
    A_c_per_class(i) = sum(A_mc_class .* facet_area_class)/groud_area;
    
    % 提取当前类别的吸收光量
    light_absorbed_class = light_absorbed(indices);
    % 计算当前类别的总吸收光量
    light_absorbed_per_class(i) = sum(light_absorbed_class);
    
    % 计算当前类别的光能利用效率
    %LUE_per_class(i) = A_c_per_class(i) / light_absorbed_per_class(i);
   % 提取当前类别的总光合速率 A_c 和每单位地面日光截获量 LI
    A_c_class = A_c_per_class(i);
    LI_class = sum(PPFD_c(indices) .* facet_area(indices)) / groud_area;
    
    % 计算当前类别的光能利用效率
    LUE_per_class(i) = A_c_class / LI_class;



end


% 输出每个类别的 A_c
%for i = 1:length(unique_labels)
 %   disp(['标签 ' num2str(unique_labels(i)) ' 光合效率: ' num2str(A_c_per_class(i))]);
  %  disp(['标签 ' num2str(unique_labels(i)) ' 吸收的光量: ' num2str(light_absorbed_per_class(i))]);
   % disp(['标签 ' num2str(unique_labels(i)) ' 光能利用效率: ' num2str(LUE_per_class(i))]);
%
%end
disp(['总值']);
disp([num2str(A_c)]);
disp([num2str(sum(PPFD_c .* facet_area) / groud_area)]);
disp([num2str(A_c / (sum(PPFD_c .* facet_area) / groud_area))]);
disp([num2str(sum(PPFD_c .* facet_area))]);
  

% 输出每个类别的 A_c
for i = 1:length(unique_labels)
    disp(['标签 ' num2str(unique_labels(i)) ])
    disp([ num2str(A_c_per_class(i))]);
    disp([ num2str(light_absorbed_per_class(i))]);
    disp([ num2str(LUE_per_class(i))]);

end

%% canopy light interception and photosynthetic rate
%result=readmatrix('E:\AQ\result.txt');
%pos=result(:,6:14);
%pos_all=[pos(:,1:3);pos(:,4:6);pos(:,7:9)];
%faces=[(1:length(pos))',(length(pos)+1:length(pos)*2)',(length(pos)*2+1:length(pos)*3)'];
%facet_area=result(:,18)/10000;
%PPFD_c=result(:,25);


%figure
%trisurf(faces,pos_all(:,1),pos_all(:,2),pos_all(:,3),PPFD_c,'FaceAlpha',0.9,'EdgeColor','none'),caxis([0 2000])
%set(gcf,'Color',[1,1,1])
%axis equal
%colorbar

% 初始化A_mc
A_mc = zeros(size(PPFD_c));

%for i=1:length(PPFD_c)
%A_mc(i) =A_Q_curve(para_fit,PPFD_c(i,1));
%end

%figure
%trisurf(faces,pos_all(:,1),pos_all(:,2),pos_all(:,3),A_mc,'FaceAlpha', 0.9,'EdgeColor','none')
%set(gcf,'Color',[1,1,1])
%axis equal
%colorbar

%disp('A_mc:');
%disp(A_mc);

%disp('facet_area:');
%disp(facet_area);

%A_hour=sum(A_mc'.*facet_area); % Total canopy photosynthetic rate
%disp('A_hour:');
%disp(A_hour);
%A_hour = sum(sum(A_mc' .* facet_area));
%disp('A_hour:');
%disp(A_hour);
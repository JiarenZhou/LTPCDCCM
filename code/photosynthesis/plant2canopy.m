function [group_rp,group_re]=plant2canopy(group_se,group_sp,label,row_num,plant_num,plant_dis,row_dis)

for i=1:plant_num
    for j=1:row_num
group_tr=coordinate_rotate(group_sp,rand(1,1)*360,[0,0,0]);       
group_cell{i,j}(:,1)=group_tr(:,1)+i*plant_dis;%%
group_cell{i,j}(:,2)=group_tr(:,2)+j*row_dis;%%加上行距
group_cell{i,j}(:,3)=group_tr(:,3);
group_cell{i,j}(:,4)=label;
    end
end
disp('group_tr:');
%disp(group_tr);
writematrix(group_tr,'group_tr.txt','Delimiter',' ');
group_rp=[];

for i=1:plant_num
    for j=1:row_num
group_rp1=group_cell{i,j};
group_rp=[group_rp;group_rp1];
    end
end
group_re=[];
n=0;
l=length(group_sp);

for i=1:plant_num
    for j=1:row_num
group_e1(:,1)=group_se(:,1)+n;
group_e1(:,2)=group_se(:,2)+n;
group_e1(:,3)=group_se(:,3)+n;
n=n+l;
group_re=[group_re;group_e1]; 
group_e1=[];
    end
end

% 在这里输出 face 和 vertex
disp('group_re:');
%disp(group_re);
writematrix(group_re,'group_re.txt','Delimiter',' ');
disp('group_rp:');
%disp(group_rp);
writematrix(group_rp,'group_rp.txt','Delimiter',' ');
end




function pos_end=coordinate_rotate(pos1,degree,zero_point)
pos_t=[pos1(:,1)-zero_point(:,1),pos1(:,2)-zero_point(:,2),pos1(:,3)-zero_point(:,3),ones(length(pos1),1)];
sin2=sind(degree);
cos2=cosd(degree);
pos_tt=pos_t*[cos2 sin2 0 0;-sin2 cos2 0 0;0 0 1 0;0 0 0 1]; 
pos_end=pos_tt(:,1:3)+zero_point;
end


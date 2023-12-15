
%{ 
old_img(i, j, k): قيمة البكسل �?ي الموقع (i, j) للقناة k �?ي الصورة القديمة.
old_min(k): القيمة الدنيا للبكسل �?ي القناة k �?ي الصورة القديمة.
old_max(k): القيمة العليا للبكسل �?ي القناة k �?ي الصورة القديمة.
new_max: القيمة العليا المرغوبة بعد تمدد الصورة.
new_min: القيمة الدنيا المرغوبة بعد تمدد الصورة.
new_img(i, j, k): قيمة البكسل الجديدة �?ي الموقع (i, j) للقناة k �?ي الصورة الم�?مددة.
%}
function [ new_img ] = contruct_stretching ( old_img,new_min,new_max )
[r,c,l]=size(old_img);
old_img=double(old_img);
old_min=min(min(old_img));
old_max=max(max(old_img));
new_img=zeros(r,c,l);
if l==3
old_min1=min(min(old_img(:,:,1)));
old_min2=min(min(old_img(:,:,2)));
old_min3=min(min(old_img(:,:,3)));

old_max1=max(max(old_img(:,:,1)));
old_max2=max(max(old_img(:,:,2)));
old_max3=max(max(old_img(:,:,3)));
end
    for i=1:r
        for j=1:c
            if l==1
                new_img(i,j,1)=((old_img(i,j,1) - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min;
            else
                new_img(i,j,1)=((old_img(i,j,1) - old_min1) / (old_max1 - old_min1)) * (new_max - new_min) + new_min;
                new_img(i,j,2)=((old_img(i,j,2) - old_min2) / (old_max2 - old_min2)) * (new_max - new_min) + new_min;
                new_img(i,j,3)=((old_img(i,j,3) - old_min3) / (old_max3 - old_min3)) * (new_max - new_min) + new_min;

            end
            
        end
    end
new_img = uint8(new_img);
imshow(new_img);
end


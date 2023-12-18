function [ new_img ] = RayLeigh_Noise( img,a,b )
[H W L]=size(img);
for i = 1:255
    pixelCount=round(((2/b)*(i-a)*(exp(((i-a).^2)/b)))*H*W);
    for j = 1 :pixelCount
        row=ceil(round(1,1)*H);
        column=ceil(round(1,1)*W);
        img(row,column)=img(row,column)+i;
    end
end
new_img=contrast_stretcing(img);
new_img=uint8(new_img);
end
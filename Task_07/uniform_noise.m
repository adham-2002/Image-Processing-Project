function [ new_img ] = uniform_noise( img, a, b )
img=double(img);
[H W L]=size(img);

pixelCount=round((1/(b-a))*H*W);

for i=1:255
    for x=1:pixelCount
        row=ceil(rand(1, 1)*H);
        column=ceil(rand(1, 1)*W);
        img(row, column)=img(row, column)+i;
    end
end
%Normalization
new_img=contrast_stretcing(img);
new_img=uint8(new_img);
end


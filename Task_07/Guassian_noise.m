function [ new_img ] = Guassian_noise( img, m, s )
img=double(img);
[H W L]=size(img);

for i=1:255
    pixelCount=round(((exp((-(i-m)^2)/(2*s^2)))/sqrt(2*3.14*s)*H*W));
    for j=1:pixelCount
        row=ceil(rand(1, 1)*H);
        column=ceil(rand(1, 1)*W);
        img(row, column)=img(row, column)+i;
    end
end
%Normalization
new_img=contrast_stretcing(img);
new_img=uint8(new_img);

end


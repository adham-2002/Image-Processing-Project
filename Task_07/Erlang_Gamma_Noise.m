function [ new_img ] = Erlang_Gamma_Noise( img,a,b )
[H W L]=size(img);
for i = 1:255
    pixelCount=round((((a.^b)*(i.^(b-1)))/(factorial(b-1)))*exp(-a*i)*H*W);
    for j = 1 :pixelCount
        row=ceil(rand(1,1)*H);
        column=ceil(rand(1,1)*W);
        img(row,column)=img(row,column)+i;
    end
end
new_img=contrast_stretcing(img);
new_img=uint8(new_img);
end
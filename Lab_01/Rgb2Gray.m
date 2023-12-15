function [gray]=Rgb2Gray(rgb,option)
[h ,w, l]=size(rgb);
gray=zeros(h,w);
gray=double(gray);
for i=1 :h
    for j=1:w
        if option==1
        gray(i,j)=(rgb(i,j,1)+rgb(i,j,2)+rgb(i,j,3))/3;
        end
        if option==2
             gray(i,j)=(rgb(i,j,1)*0.6 +rgb(i,j,2)*0.1+rgb(i,j,3)*0.3);
        end
        if option==3
            gray(i,j)=rgb(i,j,l);
        end
    end
end
gray= uint8(gray);
imshow(gray);
end
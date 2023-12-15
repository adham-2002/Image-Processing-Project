function [array] = histogram( image )
[H, W, L]=size(image);
if L == 3
    R_array=zeros(256,1);% matlab is not zero index ##
    G_array=zeros(256,1);
    B_array=zeros(256,1);
    for i = 1:H
        for j =1:W    
            R_array(image(i,j,1)+1)=R_array(image(i,j,1)+1)+1;
            G_array(image(i,j,2)+1)=G_array(image(i,j,2)+1)+1;
            B_array(image(i,j,3)+1)=B_array(image(i,j,3)+1)+1;
        end
    end
    array=[R_array,G_array,B_array];
hb = bar(array);
hb(1).FaceColor = 'r';
hb(2).FaceColor = 'g';
hb(3).FaceColor = 'b';
else
array=zeros(256,1);
    for i = 1:H
        for j =1:W    
            array(image(i,j)+1)=array(image(i,j)+1)+1;
        end
    end
    
    bar(array);
end
end


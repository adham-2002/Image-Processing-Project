## Matlab Summary
### commands in Matlab
* `imread` is used to read an image in Matlab.
* `imshow` is used to display an image in Matlab.
* `%` is used to comment in Matlab.
* `clc` is used to clear the command window.
* `clear all` is used to clear the workspace.
* `matlab\toolbox\images\imdata\circles.png` path to find the image in Matlab.
* `figure,imshow()` is used to display single or 
* `rgb2gray` is used to convert RGB image to grayscale image in Matlab.
* `im2bw` is used to convert RGB image to binary image in Matlab.
* `c(:,:,1) c(:,:,2) c(:,:,3)` is used to extract the red, green and blue channels of the image in Matlab.

* Define Function in Matlab:
```
function [output_args] = function_name(input_args)
    % Function body
    % Perform computations here
end
```
<hr>

<h4 style="
  font-family: Arial, Helvetica, sans-serif;
  background: linear-gradient(to right, #f32170, #ff6b08, #cf23cf, #eedd44);
  -webkit-text-fill-color: transparent;
  -webkit-background-clip: text;
">| Convert RGB image to grayscale </h4>

```
function [gray] = RGBTOGRAY(RGB, option)
    % Function to convert RGB image to grayscale
    
    [H, W, L] = size(RGB);
    gray = zeros(H, W);
    gray = double(gray);

    for i = 1:H
        for j = 1:W
            if option == 1
                gray(i, j) = (RGB(i, j, 1) + RGB(i, j, 2) + RGB(i, j, 3)) / 3;
            elseif option == 2
                gray(i, j) = RGB(i, j, 1) * 0.7 + RGB(i, j, 2) * 0.1 + RGB(i, j, 3) * 0.2;
            elseif option == 3
                gray(i, j) = RGB(i, j, 1);
            elseif option == 4
                gray(i, j) = RGB(i, j, 2);
            elseif option == 5
                gray(i, j) = RGB(i, j, 3);
            end
        end
    end

    gray = uint8(gray);
end
`1` Average method (R+G+B)/3.
`2` Luminance method (R0.7 + G0.1 + B*0.2).
`3` Red channel (R).
`4` Green channel (G).
`5` Blue channel (B).
```
* `size()` is used to get the size of the image in Matlab.
```
Example: 
    img=imread('peppers.png');
    [H, W, L] = size(img);
    disp(H); % 384
    disp(W); % 512
    disp(L); % 3
```
* `zeros()` is used to create an image of zeros in Matlab.
```
Example: 
    img=zeros(5,5,3);
    figure,imshow(img);
```
* `uint8` is used to create an image of unsigned 8-bit integers in Matlab because float number become zero.
```
Example: 
    img=uint8(zeros(5,5,3));
    figure,imshow(img);
```
* Convert grayscal image to binary image in Matlab:
```
function [ binary ] = GraytoBinary( gray,threshold )
[H W]=size(gray);
binary=zeros(H,W);
for i=1:H
    for j=1:W
        if gray(i,j)< threshold 
            binary(i,j)=0;
        end
        if gray(i,j)>= threshold 
            binary(i,j)=1;
        end
    end
end
binary=logical(binary);%convert 0,1 int to true and false logical 
end
```
* `logical` is used to convert 0,1 int to true and false(and or xor ) logical in Matlab.


```
function [new_image] = mylog(old_image, index)
[H, W, L] = size(old_image);
new_image = zeros(H, W);
old_image = im2double(old_image);

for i = 1:H
    for j = 1:W
        if index == 1 % Log transform => 
            if L == 3
                new_image(i, j, 1) = log(old_image(i, j, 1) + 2);
                new_image(i, j, 2) = log(old_image(i, j, 2) + 2);
                new_image(i, j, 3) = log(old_image(i, j, 3) + 2);
            else
                new_image(i, j) = log(old_image(i, j) + 2);
            end
        elseif index == 2 % Inverse log transform
            if L == 3
                new_image(i, j, 1) = exp(old_image(i, j, 1)) + 1;
                new_image(i, j, 2) = exp(old_image(i, j, 2)) + 1;
                new_image(i, j, 3) = exp(old_image(i, j, 3)) + 1;
            else
                new_image(i, j) = exp(old_image(i, j)) + 1 ;
            end
        elseif index == 3 % Power transform
            if L == 3
                new_image(i, j, 1) = old_image(i, j, 1) ^ 3;
                new_image(i, j, 2) = old_image(i, j, 2) ^ 3;
                new_image(i, j, 3) = old_image(i, j, 3) ^ 3;
            else
                new_image(i, j) = old_image(i, j) ^ 2;
            end
        elseif index == 4 % Square root transform
            if L == 3
                new_image(i, j, 1) = sqrt(old_image(i, j, 1));
                new_image(i, j, 2) = sqrt(old_image(i, j, 2));
                new_image(i, j, 3) = sqrt(old_image(i, j, 3));
            else
                new_image(i, j) = sqrt(old_image(i, j));
            end
        elseif index == 5 % Negative transform
            if L == 3
                new_image(i, j, 1) = 1 - old_image(i, j, 1);
                new_image(i, j, 2) = 1 - old_image(i, j, 2);
                new_image(i, j, 3) = 1 - old_image(i, j, 3);
            else
                new_image(i, j) = 1 - old_image(i, j);
            end
        end
    end
end
end
```
* `img2double` is used to convert image to double in Matlab.
```
function [strech_img] = stretching(img, old_min, old_max, new_min, new_max)
    [H, W, L] = size(img);
    strech_img = zeros(H, W, L);
    disp(L);
    for i = 1:H
        for j = 1:W
            for k = 1:L
                pixel_value = img(i, j, k);
                
                % Stretch the pixel value from the old range to the new range
                strech_img(i, j, k) = (pixel_value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min;
            end
        end
    end
    strech_img = uint8(strech_img);
    figure, imshow(strech_img)
end
```
``
```
function [  ] = histogram( image )
[H W L]=size(image);
if L == 3
    R_array=zeros(256,1);
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
```




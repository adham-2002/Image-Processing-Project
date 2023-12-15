function [new_image] = LogTransform(old_image)
[H, W, L] = size(old_image);
disp(L);
new_image = zeros(H, W, L);
old_image = im2double(old_image);

for i = 1:H
    for j = 1:W
        for k = 1:L
            new_image(i, j, k) = log(old_image(i, j, k) + 1);
        end
    end
end
imshow(new_image);
end

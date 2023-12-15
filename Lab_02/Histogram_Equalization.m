function [result] = Histogram_Equalization(image)
    [H, W, L] = size(image);
    result = uint8(zeros(H, W, L));

    for channel = 1:L
        array = zeros(256, 1);
        prob = zeros(256, 1);
        prob = double(prob);
        sk = zeros(256, 1);

        for i = 1:H
            for j = 1:W
                pixel_value = image(i, j, channel) + 1;
                array(pixel_value) = array(pixel_value) + 1;
                prob(pixel_value) = array(pixel_value) / (H * W);% calculate the probability for each pixel 
            end
        end

        sum = 0;
        sum = double(sum);
        for i = 1:256
            sum = sum + prob(i);
            sk(i) = 255 * (sum); 
        end

        for i = 1:H
            for j = 1:W
                result(i, j, channel) = sk(image(i, j, channel) + 1);
            end
        end
    end
end

function [ni] = Gray2Binary(a,t)
[w h]=size(a);
ni = zeros(w,h);
for i=1 : w 
    for j=1 : h
        if a(i,j)<t
            ni(i,j)=0;
        else
             ni(i,j)=1;
        end
    end
end
ni = logical(ni);
end
function valid = boxinter(a, b, min_ratio_a)
% percentage of interception wrt area of b
valid = false(size(a,1), size(b,1));
wa = a(:,3)-a(:,1)+1;
ha = a(:,4)-a(:,2)+1;
for i = 1:size(b, 1)
    x1 = max(a(:,1), b(i,1));
    y1 = max(a(:,2), b(i,2));
    x2 = min(a(:,3), b(i,3));
    y2 = min(a(:,4), b(i,4));
    w = x2-x1+1;
    h = y2-y1+1;
    valid(:,i) = all([w./wa h./ha] > min_ratio_a, 2);
end

function [out_rows, missing_inds] = get_rows(rows, inds)
out_rows = rows(inds(inds <= size(rows,1)),:);
if nargout>1
    missing_inds = inds(inds > size(rows,1));
end
end
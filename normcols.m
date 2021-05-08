function matout=normcols(matin)
if ndims(matin)==3
    sz=size(matin);
    matin=reshape(matin,sz(1)*sz(2),sz(3));
    matin=matin';
    
    l2norms = sqrt(sum(matin.*matin,1)+eps);
    matout = matin./repmat(l2norms,size(matin,1),1);
    
    matout=reshape(matout',sz(1),sz(2),sz(3));
else
    l2norms = sqrt(sum(matin.*matin,1)+eps);
    matout = matin./repmat(l2norms,size(matin,1),1);
end
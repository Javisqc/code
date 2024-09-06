function MResult = Mat(d)
%生成M矩阵
ME=diag(zeros(d,1)+1);%生成初始ME
    MResult=ME;
    for i=2:d
        MTurn=ME;
        alpha=(rand(1)-0.5)*pi*0.5;
        MTurn(1,1)=cos(alpha);
        MTurn(i,i)=cos(alpha);
        MTurn(1,i)=sin(alpha);
        MTurn(i,1)=-sin(alpha);
        MResult=MResult*MTurn;
    end
    for i=2:d-1
        MTurn=ME;
        alpha=(rand(1)-0.5)*pi*0.5;
        MTurn(i,i)=cos(alpha);
        MTurn(d,d)=cos(alpha);
        MTurn(i,d)=sin(alpha);
        MTurn(d,i)=-sin(alpha);
        MResult=MResult*MTurn;
    end
end


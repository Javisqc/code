%% LSLPSO
close all;
clear all;
clc;
%% ��������
global  popNum dim
popNum = 100;                     % ��ʼ��Ⱥ����
dim =30;
M = popNum+round(dim./10);  % ������Ⱥ����
Maxstep =5000;                    % ����������
epslon=0.01*dim./M;
pop_bound0 =[-50 50];               % ����λ�ò�������                  
record=zeros(Maxstep,dim);%��¼������Ӧ��ֵ
record2=zeros(Maxstep,dim);  %��¼���Ž�
fit=zeros(M,Maxstep); % �������Ӧ�Ⱥ���ֵ
Fit=zeros(M,Maxstep); % �����ĸ������Ӧ�Ⱥ���ֵ
PL=zeros(M,1);% ���ѧϰ����
mbest=zeros(dim,Maxstep); % ����ÿ��ά�ȵ�ƽ��ֵ
delta=zeros(M,dim,Maxstep);%ѧϰƫ����
delta2=zeros(M,dim,Maxstep);%������ѧϰƫ����
pop=zeros(M,dim,Maxstep);%��Ⱥλ��
pop2=zeros(M,dim,Maxstep);%��������Ⱥλ��
%% Cubic�����ʼ������һ�ε���λ�ã�    ע�⣺����2.595 0.3 �˾�����������
x(1)=0.3;
for i=2:(M*dim)
x(i)=2.595.*x(i-1).*(1-x(i-1).^2);  %Cubic
end
A=reshape(x,[M,dim]);
pop(:,:,1)=pop_bound0(1)+A*(pop_bound0(2)-pop_bound0(1));
%�������������ʼ����������
% pop(:,:,1) = pop_bound0(1)+rand(M,dim,1)*(pop_bound0(2)-pop_bound0(1));%��ʼ��Ⱥ��λ��(������������趨
%��������������������������
for i = 1:M
    PL(i)=(1-((i-1)./M)).^(0.5*(log(ceil(dim./popNum))));% �������ѧϰ����
end
%% �Ŷ���ֵ�趨
Tk=50;
Tc=50;
r4=1;
r5=1;
TK=1;
TC=1;
%  sigmaԭʼ���㹫ʽ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:M
        fit(i,1) = f(pop(i,:,1));    % �������Ӧ�Ⱥ���ֵ
    end   
    [Fit(:,1),K] = sort(fit(:,1),'descend'); % �Ե�һ�ε�������Ӧ�Ⱥ���ֵ��������
    record(1,1)=min(Fit(:,1));%��һ�ε�������С��Ӧ��ֵ
    
    bbb=record(1,1);
    pop_bbb=pop(K(M),:,1);
    aaa=fit(M,1);
    ccc=pop(K(M),:,1);
    fit_ccc = min(Fit(:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ��������   ��ѭ��   ����������������
for step = 2:Maxstep+1
    
    % -----------����һ�ε���������λ�ø�����Ӧ��ֵ����-----------
    for i = 1:M
        fit(i,step-1) = f(pop(i,:,step-1));    % �������Ӧ�Ⱥ���ֵ
    end   
    [Fit(:,step-1),K] = sort(fit(:,step-1),'descend'); % �Ե�һ�ε�������Ӧ�Ⱥ���ֵ��������
    record(step-1,1)=min(Fit(:,step-1));%��һ�ε�������С��Ӧ��ֵ
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Fit(M,step-1) < fit_ccc    %��֤��ccc����С
        fit_ccc = Fit(M,step-1);
        ccc = pop(K(M),:,step-1);
    else
        Fit(M,step-1)=fit_ccc;
        pop(K(M),:,step-1) = ccc;
    end
    record(step-1,1)=Fit(M,step-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i=1:M
        pop2(i,:,step-1)=pop(K(i),:,step-1);%������Ӧ�Ⱥ����������¶Ե�һ�ε�������Ⱥλ������
        delta2(i,:,step-1)=delta(K(i),:,step-1);%������Ӧ�Ⱥ����������¶Ե�һ�ε�������Ⱥλ������
    end  
    record2(step-1,:)=pop2(M,:,step-1)';%��һ�ε��������Ž�

    for j=1:dim
    mbest(j,step-1) =sum(pop2(:,j,step-1))/M;        %��һ�ε���������ÿ��ά�ȵ�ƽ��ֵ
    end 
    
    % -----------���ε�������������-----------
    r1= rand;
    r2= 0.8;   %0.8��ʱ��ȽϺ�
    r3= rand;
    %________���ε������Ŷ�����__________
     if step>2
         if pop2(M,:,step-1)==pop2(M,:,step-2)
             TK=TK+1;
         else
             TK=1;
         end
         if  mbest(:,step-1)==mbest(:,step-2)
             TC=TC+1;
         else
             TC=1;
         end
     end
     r4=rand*(TK>Tk)+1*(TK<=Tk);
     r5=rand*(TC>Tc)+1*(TC<=Tc);
    % -----------����λ�ø���-----------
     for i = 1:M-1
            p=rand;   %���ѧϰ����
            if p<=PL(i)  %С�����ѧϰ����ʱѧϰ
                for j = 1:dim
                    jy=randi([i+1,M]); %�ϲ�������jy������ѧϰ
                    delta(i,j,step)= r1.*delta2(i,j,step-1)+r2.*(r4*pop2(jy,j,step-1)-pop2(i,j,step-1))+epslon.*r3.*(r5*mbest(j,step-1)-pop2(i,j,step-1));%ѧϰƫ����
                    pop(i,j,step)=pop2(i,j,step-1)+ delta(i,j,step);
                end
            else %�������ѧϰ����ʱ��ѧϰ
                    pop(i,:,step)=pop2(i,:,step-1);
            end
     end
     %�����������������ӵĲ���̰��levy���л��ơ���������
     beta1=1.5;
     sigma1 =(gamma(1+beta1)*sin(beta1*pi/2)./gamma(0.5+0.5*beta1)./beta1./(2^(0.5*beta1-0.5)))^(1/beta1);
     delta1 = normrnd(0,sigma1,[1,dim]);    %��̬�����
     sita1 = normrnd(0,1,[1,dim]);
     Levy_step=sigma1./(abs(delta1).^(1./beta1));
     gbest=pop2(M,:,step-1);
     glast=pop2(1,:,step-1);
     gbest_levy=gbest+randi(dim).*(gbest-glast).*Levy_step;
     if f(gbest_levy)<f(glast)
         if (sum(gbest_levy>pop_bound0(1))==dim)&&(sum(gbest_levy(:)<pop_bound0(2))==dim)
             pop2(M,:,step)=gbest_levy;
         end
     end 
     
     %����һ�ε���������ֵ�������ε������һ������
     pop(M,:,step)=pop2(M,:,step-1);
     %����������
     for i = 1:M
         pop(i,:,step)=min(pop(i,:,step),pop_bound0(2));
         pop(i,:,step)=max(pop(i,:,step),pop_bound0(1));
     end
      
     delta(M,:,step)=delta2(M,:,step-1);
end

%-------- plot_draw -----------
figure(1)
plot(record(:,1),'r-*','linewidth',1);
title('��������');
axis tight
figure(2)

semilogy(record(:,1),'r-*','linewidth',1);
title('��������');
axis tight
% function y=f(x)   %sphere����
% global dim      %����ά�� [-100 50]
% y=0;
% for k=1:dim
%     mx=x(k).^2;
%     y=mx+y;
% end
% y=y;    %������С�Ӹ���
% end
record(:,1)
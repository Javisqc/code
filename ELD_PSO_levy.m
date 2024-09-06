%% LSLPSO
close all;
clear all;
clc;
%% 参数设置
global  popNum dim
popNum = 100;                     % 初始种群个数
dim =30;
M = popNum+round(dim./10);  % 给定种群个数
Maxstep =5000;                    % 最大迭代次数
epslon=0.01*dim./M;
pop_bound0 =[-50 50];               % 设置位置参数限制                  
record=zeros(Maxstep,dim);%记录最优适应度值
record2=zeros(Maxstep,dim);  %记录最优解
fit=zeros(M,Maxstep); % 个体的适应度函数值
Fit=zeros(M,Maxstep); % 排序后的个体的适应度函数值
PL=zeros(M,1);% 社会学习概率
mbest=zeros(dim,Maxstep); % 粒子每个维度的平均值
delta=zeros(M,dim,Maxstep);%学习偏移量
delta2=zeros(M,dim,Maxstep);%排序后的学习偏移量
pop=zeros(M,dim,Maxstep);%种群位置
pop2=zeros(M,dim,Maxstep);%排序后的种群位置
%% Cubic混沌初始化（第一次迭代位置）    注意：参数2.595 0.3 乃经验数据所得
x(1)=0.3;
for i=2:(M*dim)
x(i)=2.595.*x(i-1).*(1-x(i-1).^2);  %Cubic
end
A=reshape(x,[M,dim]);
pop(:,:,1)=pop_bound0(1)+A*(pop_bound0(2)-pop_bound0(1));
%————随机初始化————
% pop(:,:,1) = pop_bound0(1)+rand(M,dim,1)*(pop_bound0(2)-pop_bound0(1));%初始种群的位置(在区间内随机设定
%—————————————
for i = 1:M
    PL(i)=(1-((i-1)./M)).^(0.5*(log(ceil(dim./popNum))));% 计算社会学习概率
end
%% 扰动阈值设定
Tk=50;
Tc=50;
r4=1;
r5=1;
TK=1;
TC=1;
%  sigma原始计算公式

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:M
        fit(i,1) = f(pop(i,:,1));    % 个体的适应度函数值
    end   
    [Fit(:,1),K] = sort(fit(:,1),'descend'); % 对第一次迭代的适应度函数值进行排序
    record(1,1)=min(Fit(:,1));%上一次迭代的最小适应度值
    
    bbb=record(1,1);
    pop_bbb=pop(K(M),:,1);
    aaa=fit(M,1);
    ccc=pop(K(M),:,1);
    fit_ccc = min(Fit(:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ————   主循环   ————————
for step = 2:Maxstep+1
    
    % -----------对上一次迭代的粒子位置根据适应度值排序-----------
    for i = 1:M
        fit(i,step-1) = f(pop(i,:,step-1));    % 个体的适应度函数值
    end   
    [Fit(:,step-1),K] = sort(fit(:,step-1),'descend'); % 对第一次迭代的适应度函数值进行排序
    record(step-1,1)=min(Fit(:,step-1));%上一次迭代的最小适应度值
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Fit(M,step-1) < fit_ccc    %保证了ccc是最小
        fit_ccc = Fit(M,step-1);
        ccc = pop(K(M),:,step-1);
    else
        Fit(M,step-1)=fit_ccc;
        pop(K(M),:,step-1) = ccc;
    end
    record(step-1,1)=Fit(M,step-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i=1:M
        pop2(i,:,step-1)=pop(K(i),:,step-1);%根据适应度函数排列重新对第一次迭代的种群位置排列
        delta2(i,:,step-1)=delta(K(i),:,step-1);%根据适应度函数排列重新对第一次迭代的种群位置排列
    end  
    record2(step-1,:)=pop2(M,:,step-1)';%上一次迭代的最优解

    for j=1:dim
    mbest(j,step-1) =sum(pop2(:,j,step-1))/M;        %上一次迭代中粒子每个维度的平均值
    end 
    
    % -----------本次迭代的收缩因子-----------
    r1= rand;
    r2= 0.8;   %0.8的时候比较好
    r3= rand;
    %________本次迭代的扰动因子__________
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
    % -----------进行位置更新-----------
     for i = 1:M-1
            p=rand;   %随机学习概率
            if p<=PL(i)  %小于社会学习概率时学习
                for j = 1:dim
                    jy=randi([i+1,M]); %较差个体向第jy个个体学习
                    delta(i,j,step)= r1.*delta2(i,j,step-1)+r2.*(r4*pop2(jy,j,step-1)-pop2(i,j,step-1))+epslon.*r3.*(r5*mbest(j,step-1)-pop2(i,j,step-1));%学习偏移量
                    pop(i,j,step)=pop2(i,j,step-1)+ delta(i,j,step);
                end
            else %大于社会学习概率时不学习
                    pop(i,:,step)=pop2(i,:,step-1);
            end
     end
     %—————最优粒子的采用贪婪levy飞行机制—————
     beta1=1.5;
     sigma1 =(gamma(1+beta1)*sin(beta1*pi/2)./gamma(0.5+0.5*beta1)./beta1./(2^(0.5*beta1-0.5)))^(1/beta1);
     delta1 = normrnd(0,sigma1,[1,dim]);    %正态随机数
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
     
     %将上一次迭代的最优值赋给本次迭代最后一个数据
     pop(M,:,step)=pop2(M,:,step-1);
     %上下限限制
     for i = 1:M
         pop(i,:,step)=min(pop(i,:,step),pop_bound0(2));
         pop(i,:,step)=max(pop(i,:,step),pop_bound0(1));
     end
      
     delta(M,:,step)=delta2(M,:,step-1);
end

%-------- plot_draw -----------
figure(1)
plot(record(:,1),'r-*','linewidth',1);
title('收敛过程');
axis tight
figure(2)

semilogy(record(:,1),'r-*','linewidth',1);
title('收敛过程');
axis tight
% function y=f(x)   %sphere函数
% global dim      %设置维度 [-100 50]
% y=0;
% for k=1:dim
%     mx=x(k).^2;
%     y=mx+y;
% end
% y=y;    %最大变最小加负号
% end
record(:,1)
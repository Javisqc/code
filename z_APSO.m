
%标准PSO---Acley函数---仿真
%%参数初始化
    clc;
    clear;
    clear all
    c1=1.49445;
    c2=1.49445;
    maxg=5000;     %进化次数
    %--------------------------------------------------------------------------
    sizepop=100;    %种群规模  N=20/80
    par_num=30;     %粒子维度  D=10/30
    popmax=5.12;     %种群上下边界值
    %  32      5.12       600       10      100     30          10      
    %  Ackley  Rastrigin  Griewank  Alpine  Sphere  Rosenbrock  Schwefel
    %
    %  1
    %  Sum_of_Different_Power
    %--------------------------------------------------------------------------
    popmin=-popmax;
    Vmax=0.15*popmax;
    Vmin=0.15*popmin; 


    wmax=0.8;
    wmin=0.6;

%重复t=50次
for t=1:1
    %%产生初始粒子和速度
    for i=1:sizepop
        pop(i,:)=popmax.*rands(1,par_num);    %初始位置
        V(i,:)=Vmax.*rands(1,par_num);        %初始速度
        fitness(i)=f(pop(i,:));%适应度------------------------------
    end

    %寻找最优个体
    [bestfitness bestindex]=min(fitness);
    pBest=pop;                  %个体最佳
    gBest=pop(bestindex,:);     %全局最佳
    fitnesspbest=fitness;       %个体最佳适应度
    fitnessgbest=bestfitness;   %全局最佳适应度

    %%迭代寻优
    for i=1:maxg
        for j=1:sizepop
            fv(j)=f(pop(j,:));
        end
        favg=sum(fv)/sizepop;
        fmin=min(fv);
        for j=1:sizepop
            if fv(j)<=favg
                w= wmin+(fv(j)-fmin)*(wmax-wmin)/(favg-fmin);
            else
                w=wmax;
            end
            %速度更新
            V(j,:)=w*V(j,:)+c1*rand*(pBest(j,:)-pop(j,:))+c2*rand*(gBest-pop(j,:));
            V(j,find(V(j,:)>Vmax))=Vmax;
            V(j,find(V(j,:)<Vmin))=Vmin;

            %种群更新
            pop(j,:)=pop(j,:)+V(j,:);
            pop(j,find(pop(j,:)>popmax))=popmax;
            pop(j,find(pop(j,:)<popmin))=popmin;

    %         %自适应变异
    %         if rand>0.8
    %             k=ceil(par_num*rand);
    %             pop(j,k)=rand;
    %         end

    %         %适应度值
                     fitness(j)=f(pop(j,:));%------------------------------------

            %个体最优更新
            if fitness(j)<fitnesspbest(j)
                pBest(j,:)=pop(j,:);
                fitnesspbest(j)=fitness(j);
            end

            %群体最优更新
            if fitness(j)<fitnessgbest
                gBest=pop(j,:);
                fitnessgbest=fitness(j);
            end
        end
        result(i)=fitnessgbest;
    end

%     plot(result);
%     title('适应度曲线 ');
%     grid on
%     xlabel('进化代数');
%     ylabel('适应度');
time(t)=result(1000);
end
figure(1)
semilogy(result,'r-*','linewidth',2);
title('收敛过程');
axis tight
figure(2)
plot(result,'r-*','linewidth',2);
title('收敛过程');
axis tight
 result'   
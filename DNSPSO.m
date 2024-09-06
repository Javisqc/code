%DNSPSO--仿真
%%参数初始化
clc;
clear;
clear all
global dim
Neighbor_num=4;   %%选择粒子最优邻域的数量
maxg=5000;     %进化次数
%--------------------------------------------------------------------------
sizepop=100;    %种群规模  N=20/80
par_num=30;     %粒子维度  D=10/30
dim=par_num;
popmax=1.28;     %种群上下边界值
%  32      5.21       600       10      100     30          10
%  Ackley  Rastrigin  Griewank  Alpine  Sphere  Rosenbrock  Schwefel
%
%  1
%  Sum_of_Different_Power
%--------------------------------------------------------------------------
%初始化范围
popmin = -popmax;

 Vmax = 0.15*popmax;
Vmin = 0.15*popmin;

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
    for i=1:maxg     %迭代寻优次数
        CR=0.4+0.5*i./maxg;
        F=0.5+0.5*i./maxg;
      %————--计算粒子距离—————
        for j=1:sizepop   %第j个粒子
            POP=((pop(j,:)-pop)*(pop(j,:)-pop)').^0.5;
            d(j)=(1./(par_num-1))*trace(POP);     %由此得到粒子的计算状态
        end
             %——————计算全局最优的附近位置————————
            D_gb=((pop-gBest)*(pop-gBest)').^0.5;
            B=diag(D_gb);
            [d_gb_sort,K] = sort(B,'ascend');
            rand_gb=randi([1,Neighbor_num]);   %随机选择全局最优邻域
            gBest_Neighbor=pop(rand_gb);
        %% —————迭代开始—————————
        for j=1:sizepop   %第j个粒子
%             j
            %—————计算每个局部最优粒子周围的粒子位置——————
            D_pb=((pop-pBest(j,:))*(pop-pBest(j,:))').^0.5;
            A=diag(D_pb);
            [aa,K] = sort(D_pb,'ascend');
            rand_pb=randi([1,Neighbor_num]);   %随机选择粒子最优邻域，第一个是自身，加1
            pBest_Neighbor=pop(K(rand_pb),:);   %得到pBest的neighbor
            %————--计算粒子距离—————
            Ef(j)=(d(j)-min(d))./(max(d)-min(d));    %进化因子
            %—————自适应权重——————
            w=0.5*Ef(j)+0.4;   %惯性权重
            %———————四种模式——————
                if (Ef(j)<0.25)
                    if rand<0.9
                        sigma=1;    %收敛状态
                        pBest1(j,:)=pBest(j,:);
                        gBest1=gBest;
                        c1=2;
                        c2=2;
                    else
                        sigma=2;
                        pBest1(j,:)=pBest_Neighbor;
                        gBest1=gBest;
                        c1=2.1;
                        c2=1.9;
                    end
                elseif ((Ef(j)>=0.25)&&(Ef(j)<0.5))
                    aa=rand;
                    if aa<0.9
                        sigma=2;    %开发状态
                        pBest1(j,:)=pBest_Neighbor;
                        gBest1=gBest;
                        c1=2.1;
                        c2=1.9;
                    elseif aa>=0.95
                        sigma=1;
                        pBest1(j,:)=pBest(j,:);
                        gBest1=gBest;
                        c1=2;
                        c2=2;
                    else
                        sigma=3;
                        pBest1(j,:)=pBest(j,:);
                        gBest1=gBest_Neighbor;
                        c1=2.2;
                        c2=1.8;
                    end
                elseif ((Ef(j)>=0.5)&&(Ef(j)<0.75))
                    aaa=rand;
                    if aaa<0.9
                        sigma=3;    %探索状态
                        pBest1(j,:)=pBest(j,:);
                        gBest1=gBest_Neighbor;
                        c1=2.2;
                        c2=1.8;
                    elseif aaa>=0.95
                        sigma=2;
                        pBest1(j,:)=pBest_Neighbor;
                        gBest1=gBest;
                        c1=2.1;
                        c2=1.9;
                    else
                        sigma=4;
                        pBest1(j,:)=pBest_Neighbor;
                        gBest1=gBest_Neighbor;
                        c1=1.8;
                        c2=2.2;
                    end
                elseif (Ef(j)>=0.75)
                    if rand>0.9
                        sigma=4;    %跳出状态
                        pBest1(j,:)=pBest_Neighbor;
                        gBest1=gBest_Neighbor;
                        c1=1.8;
                        c2=2.2;
                    else
                        sigma=3; 
                        pBest1(j,:)=pBest(j,:);
                        gBest1=gBest_Neighbor;
                        c1=2.2;
                        c2=1.8;
                    end
            end
            %—————————差分进化—————————————
            
            V(j,:)=pBest(j,:)+F.*(ones(1,30)).*(pBest(randi(sizepop),:)-pBest(randi(sizepop),:));
            u(j,:)=V(j,:)*(rand<=CR)+pBest(j,:);
            
            V(j,:)=w*V(j,:)+c1*rand*(pBest1(j,:)-pop(j,:))+c2*rand*(gBest1-pop(j,:));
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
%     time(t)=result(1000);
end
    figure(1)
    semilogy(result,'r-*','linewidth',2);
    title('收敛过程');
    axis tight
    % figure(2)
    % plot(result,'r-*','linewidth',2);
    % title('收敛过程');
    % axis tight
result'
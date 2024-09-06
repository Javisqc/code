
%����      J. Kennedy, ��Bare bones particle swarms,�� in Proc. IEEE Swarm Intell.
% Symp., Apr. 2003, pp. 80�C87
%%������ʼ��
format long;
clc;
clear all;
maxg=5000;    %��������

%alter
sizepop=100;  %��Ⱥ��ģ-------------------N=20/80
D=30;        %ά��-----------------------D=10/30
popmax=600;    %��Ⱥ���±߽�ֵ-------------------------------------------------------��

%  32      5.21       600       10      100     30          10      
%  Ackley  Rastrigin  Griewank  Alpine  Sphere  Rosenbrock  Schwefel
%
%  1
%  Sum_of_Different_Power
popmin=-popmax;
Vmax=0.15*popmax;
Vmin=0.15*popmin;



%�ظ�50��
for t=1:1
    %%��ʼ����Ⱥ
    for i=1:sizepop
        pop(i,:)=popmax.*rands(1,D);    %��ʼλ��
        %V(i,:)=Vmax.*rands(1,D);        %��ʼ�ٶ�
        fitness(i)=f(pop(i,:));%��Ӧ��-----------------------------------------��
    end
    
    %Ѱ�����Ÿ���
    pBest=pop;                  %�������
    [bestfitness bestindex]=min(fitness);
    gBest=pop(bestindex,:);     %ȫ�����
    
    fitnesspbest=fitness;       %���������Ӧ��
    fitnessgbest=bestfitness;   %ȫ�������Ӧ��
    
    %%����Ѱ��
    for i=1:maxg
        for j=1:sizepop            
            %��Ⱥ����
            %pop(j,:)=pop(j,:)+V(j,:);
            MU=0.5*(pBest(j,:)+gBest);
            SIGMA=abs(pBest(j,:)-gBest);
            
            %����������������GAIGAIGAI��������������������
%             nu=2*i; 
%             lambda = gamrnd(nu/2,nu/2);
%             SIGMA = SIGMA/lambda;
%             pop(j,:) = (MU+SIGMA.*(randn(1,D)))';
            i
            
              pop(j,:)=normrnd(MU,SIGMA,[1,D]);
            pop(j,find(pop(j,:)>popmax))=popmax;
            pop(j,find(pop(j,:)<popmin))=popmin;

            %%��Ӧ��ֵ
            fitness(j)=f(pop(j,:));%---------------------------------------------��
            
            %�������Ÿ���
            if fitness(j)<fitnesspbest(j)
                pBest(j,:)=pop(j,:);
                fitnesspbest(j)=fitness(j);
            end
            
            %Ⱥ�����Ÿ���
            if fitness(j)<fitnessgbest
                gBest=pop(j,:);
                fitnessgbest=fitness(j);
            end
        end
        result(i)=fitnessgbest;
    end
    time(t)=result(maxg);
end

figure(1)
semilogy(result,'r-*','linewidth',2);
title('��������');
axis tight
figure(2)
plot(result,'r-*','linewidth',2);
title('��������');
axis tight

result';












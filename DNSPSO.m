%DNSPSO--����
%%������ʼ��
clc;
clear;
clear all
global dim
Neighbor_num=4;   %%ѡ�������������������
maxg=5000;     %��������
%--------------------------------------------------------------------------
sizepop=100;    %��Ⱥ��ģ  N=20/80
par_num=30;     %����ά��  D=10/30
dim=par_num;
popmax=1.28;     %��Ⱥ���±߽�ֵ
%  32      5.21       600       10      100     30          10
%  Ackley  Rastrigin  Griewank  Alpine  Sphere  Rosenbrock  Schwefel
%
%  1
%  Sum_of_Different_Power
%--------------------------------------------------------------------------
%��ʼ����Χ
popmin = -popmax;

 Vmax = 0.15*popmax;
Vmin = 0.15*popmin;

%�ظ�t=50��
for t=1:1
    %%������ʼ���Ӻ��ٶ�
    for i=1:sizepop
        pop(i,:)=popmax.*rands(1,par_num);    %��ʼλ��
        V(i,:)=Vmax.*rands(1,par_num);        %��ʼ�ٶ�
        fitness(i)=f(pop(i,:));%��Ӧ��------------------------------
    end
    %Ѱ�����Ÿ���
    [bestfitness bestindex]=min(fitness);
    pBest=pop;                  %�������
    gBest=pop(bestindex,:);     %ȫ�����
    fitnesspbest=fitness;       %���������Ӧ��
    fitnessgbest=bestfitness;   %ȫ�������Ӧ��
    %%����Ѱ��
    for i=1:maxg     %����Ѱ�Ŵ���
        CR=0.4+0.5*i./maxg;
        F=0.5+0.5*i./maxg;
      %��������--�������Ӿ��롪��������
        for j=1:sizepop   %��j������
            POP=((pop(j,:)-pop)*(pop(j,:)-pop)').^0.5;
            d(j)=(1./(par_num-1))*trace(POP);     %�ɴ˵õ����ӵļ���״̬
        end
             %����������������ȫ�����ŵĸ���λ�á���������������
            D_gb=((pop-gBest)*(pop-gBest)').^0.5;
            B=diag(D_gb);
            [d_gb_sort,K] = sort(B,'ascend');
            rand_gb=randi([1,Neighbor_num]);   %���ѡ��ȫ����������
            gBest_Neighbor=pop(rand_gb);
        %% ����������������ʼ������������������
        for j=1:sizepop   %��j������
%             j
            %��������������ÿ���ֲ�����������Χ������λ�á�����������
            D_pb=((pop-pBest(j,:))*(pop-pBest(j,:))').^0.5;
            A=diag(D_pb);
            [aa,K] = sort(D_pb,'ascend');
            rand_pb=randi([1,Neighbor_num]);   %���ѡ�������������򣬵�һ����������1
            pBest_Neighbor=pop(K(rand_pb),:);   %�õ�pBest��neighbor
            %��������--�������Ӿ��롪��������
            Ef(j)=(d(j)-min(d))./(max(d)-min(d));    %��������
            %��������������ӦȨ�ء�����������
            w=0.5*Ef(j)+0.4;   %����Ȩ��
            %������������������ģʽ������������
                if (Ef(j)<0.25)
                    if rand<0.9
                        sigma=1;    %����״̬
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
                        sigma=2;    %����״̬
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
                        sigma=3;    %̽��״̬
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
                        sigma=4;    %����״̬
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
            %��������������������ֽ�����������������������������
            
            V(j,:)=pBest(j,:)+F.*(ones(1,30)).*(pBest(randi(sizepop),:)-pBest(randi(sizepop),:));
            u(j,:)=V(j,:)*(rand<=CR)+pBest(j,:);
            
            V(j,:)=w*V(j,:)+c1*rand*(pBest1(j,:)-pop(j,:))+c2*rand*(gBest1-pop(j,:));
            V(j,find(V(j,:)>Vmax))=Vmax;
            V(j,find(V(j,:)<Vmin))=Vmin;
            
            %��Ⱥ����
            pop(j,:)=pop(j,:)+V(j,:);
            pop(j,find(pop(j,:)>popmax))=popmax;
            pop(j,find(pop(j,:)<popmin))=popmin;
            
            %         %����Ӧ����
            %         if rand>0.8
            %             k=ceil(par_num*rand);
            %             pop(j,k)=rand;
            %         end
            
            %         %��Ӧ��ֵ
            fitness(j)=f(pop(j,:));%------------------------------------
            
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
    
    %     plot(result);
    %     title('��Ӧ������ ');
    %     grid on
    %     xlabel('��������');
    %     ylabel('��Ӧ��');
%     time(t)=result(1000);
end
    figure(1)
    semilogy(result,'r-*','linewidth',2);
    title('��������');
    axis tight
    % figure(2)
    % plot(result,'r-*','linewidth',2);
    % title('��������');
    % axis tight
result'
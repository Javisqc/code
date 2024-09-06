clear all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  !!!!注意报错需要改变初始粒子x的转置
%            在f文件里面修改：‘：
%2022.12.16  修改后可以直接运行
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [gbest,gbestval,fitcount,suc,suc_fes]
% XPSO(jingdu,func_num,fhd,Dimension,Particle_Number,Max_Gen,Max_FES,VRmin,VRmax,varargin)

global fbias
jingdu=0;
func_num=1;
D=30;
Xmax = 5.12;
Xmin=-Xmax;
fes_max=10000*D;

Particle_Number = 100;
Max_Gen=5000;
Dimension=30;

% CEC2013
fbias=[-1400, -1300, -1200, -1100, -1000,...
          -900,  -800,  -700,  -600,  -500,...
          -400,  -300,  -200,  -100,   100,...
          200,    300,   400,   500,   600,...
          700,    800,   900,  1000,  1100,...
          1200,   1300,  1400 ];
      

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
global orthm best_f best_keep initial_flag fbias gbias                                                  
%[gbest,gbestval,fitcount]= PSO_local_func('f8',3500,200000,30,30,-5.12,5.12)
rand('state',sum(100*clock));
me=Max_Gen;
ps=Particle_Number;
D=Dimension;
VRmin = Xmin;
VRmax = Xmax;
Max_FES=fes_max;
cc=[1.0 0.5 0.5];   %acceleration constants
iwt=0.9-(1:me).*(0.5/me);

neighbor(1,:)=[ps,2];
for i=2:(ps-1)
    neighbor(i,:)=[i-1,i+1];
end
neighbor(ps,:)=[ps-1,1];

if length(VRmin)==1
    VRmin=repmat(VRmin,1,D);
    VRmax=repmat(VRmax,1,D);
end
mv=0.2.*(VRmax-VRmin);
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);
Vmin=repmat(-mv,ps,1);
Vmax=-Vmin;
pos=VRmin+(VRmax-VRmin).*rand(ps,D);   %初始粒子
for i=1:ps;
e(i,1)=f(pos(i,:))
end


vel=Vmin+2.*Vmax.*rand(ps,D);%initialize the velocity of the particles
% vel=zeros(ps,D);

pbest=pos;
pbestval=e; %initialize the pbest and the pbest's fitness value
[gbestval,gbestid]=min(pbestval);

gbest=pbest(gbestid,:);%initialize the gbest and the gbest's fitness value
gbestrep=repmat(gbest,ps,1);
g_res(1)=gbestval;

recorded = 0; 
suc = 0;
suc_fes = 0;
fitcount=ps;
%%%% 
old = 1;
new = fitcount;
yyy(old:new) = min(gbestval);
old = new;

elite_ratio=0.5;
ps_elite=floor(ps*elite_ratio);  % 0.2

max_forget_ratio=0.6;  % 除了elite个体外的最差个体遗忘的维数(对gbest的)
min_forget_ratio=0.3;  % 除了elite个体外的最优个体遗忘的维数(对gbest的)
max_forget_ratio2=0.6; % 除了elite个体外的最差个体遗忘的维数(对pbest的)
min_forget_ratio2=0.3; % 除了elite个体外的最优个体遗忘的维数(对pbest的)

for i=1:ps
    dis(i)=norm(pbest(i,:)-gbestrep(i,:));
end
[Val Index]=sort(dis,'ascend');  %%%   文章采用的
% [Val Index]=sort(e,'ascend');  %%%   自己新测试的:适应值越高，越难遗忘
forget_Num(1:ps,1)=0; forget_Num2(1:ps,1)=0; % 遗忘维数
forget_Dim=zeros(ps,D); forget_Dim2=zeros(ps,D);
pmiu(1:ps,1)=0; pmiu2(1:ps,1)=0;       % 遗忘程度

factor=0.01;  % 搜索空间的加权值

%  个体对 gbest 的遗忘
pmiu(Index(1:ps_elite),1)=0;
pmiu(Index(ps_elite+1:ps),1)=(log(ps_elite+1:ps)/(2*ps));  % *(1-fitcount/Max_FES)); %     
forget_Num(Index(ps_elite+1:ps),1)=floor(D*(min_forget_ratio+(max_forget_ratio-min_forget_ratio).*((1:ps-ps_elite)./(ps-ps_elite))));
for k=1:ps
    Index1=randperm(D);
    forget_Dim(k,Index1(1:forget_Num(k)))=1;
end
pmiu(:,1:D)=repmat(pmiu(:,1),1,D);
radius=max(pos)-min(pos);
radius=repmat(radius,ps,1);
pmiu(:,:)=pmiu(:,:).*radius*factor;       
pmiu=pmiu.*forget_Dim;

%  个体对 pbest 的遗忘
pmiu2(Index(1:ps_elite),1)=0;
pmiu2(Index(ps_elite+1:ps),1)=(log(ps_elite+1:ps)/(2*ps));  % *(1-fitcount/Max_FES)); %     
forget_Num2(Index(ps_elite+1:ps),1)=floor(D*(min_forget_ratio2+(max_forget_ratio2-min_forget_ratio2).*((1:ps-ps_elite)./(ps-ps_elite))));
for k=1:ps
    Index2=randperm(D);
    forget_Dim2(k,Index2(1:forget_Num2(k)))=1;
end
pmiu2(:,1:D)=repmat(pmiu2(:,1),1,D);
radius2=max(pos)-min(pos);
radius2=repmat(radius2,ps,1);
pmiu2(:,:)=pmiu2(:,:).*radius2*factor;   

% pmiu2=pmiu2.*forget_Dim2; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

pbest_stop=zeros(1,ps);
pbest_improve=zeros(1,ps);
gbest_stop=0;
    
 cmiu1=1.05*ones(ps,1);%1.35*ones(ps,D); 
%cmiu1=2*rand(ps,D);
low1=0.5; up1=2.05;
 cmiu2=1.05*ones(ps,1);%1.35*ones(ps,D);
%cmiu2=2*rand(ps,D);
low2=0.5; up2=2.05;
 cmiu3=1.05*ones(ps,1);%1.35*ones(ps,D);
%cmiu3=2*rand(ps,D);
low3=0.5; up3=2.05;


delta=0.1;

% c1=ones(ps,D);
% c2=miu+delta*randn(ps,D);
% c3=miu+delta*randn(ps,D);
[c1, c2, c3] = randFCR(ps, D, cmiu1, low1, up1, cmiu2, low2, up2, cmiu3, low3, up3, delta); 

weight = 0.10;
n_elite=floor(elite_ratio*ps);
ww=log((ps + 1)/2) - log(1:n_elite)';
ww=repmat(ww,1,D);
cycle=10;  
kk=1;
for i=2:me
           
    if mod(i,cycle)==0 || gbest_stop>=cycle/2  || sum(pbest_stop>=cycle/2)>ps/2

        if gbest_stop>=cycle/2           
           
            cmiu1 = (1 - weight) * cmiu1 + weight * repmat(mean(good_cmiu1),ps,1);
            cmiu2 = (1 - weight) * cmiu2 + weight * repmat(mean(good_cmiu2),ps,1);
            cmiu3 = (1 - weight) * cmiu3 + weight * repmat(mean(good_cmiu3),ps,1);
            [c1, c2, c3] = randFCR(ps, D, cmiu1, low1, up1, cmiu2, low2, up2, cmiu3, low3, up3, delta); 
       
        end
    
        for i=1:ps
            dis(i)=norm(pbest(i,:)-gbestrep(i,:));
        end
        [Val Index]=sort(dis,'ascend');
%         [Val Index]=sort(e,'ascend');
        forget_Num(1:ps,1)=0;  % 遗忘维数
        forget_Dim=zeros(ps,D);
        pmiu(1:ps,1)=0;        % 遗忘程度

        %  个体对 gbest 的遗忘
        pmiu(Index(1:ps_elite),1)=0;
        pmiu(Index(ps_elite+1:ps),1)=(log(ps_elite+1:ps)/(2*ps));  % *(1-fitcount/Max_FES)); %    
        forget_Num(Index(ps_elite+1:ps),1)=floor(D*(min_forget_ratio+(max_forget_ratio-min_forget_ratio).*((1:ps-ps_elite)./(ps-ps_elite))));
        
        for k=1:ps
            Index1=randperm(D);
            forget_Dim(k,Index1(1:forget_Num(k)))=1;
        end

        pmiu(:,1:D)=repmat(pmiu(:,1),1,D);
        radius=max(pos)-min(pos);
        radius=repmat(radius,ps,1);
        pmiu(:,:)=pmiu(:,:).*radius*factor;      
%         pmiu=pmiu.*forget_Dim; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
        
        %  个体对 pbest 的遗忘
        pmiu2(Index(1:ps_elite),1)=0;
        pmiu2(Index(ps_elite+1:ps),1)=(log(ps_elite+1:ps)/(2*ps));  % *(1-fitcount/Max_FES)); %     
        forget_Num2(Index(ps_elite+1:ps),1)=floor(D*(min_forget_ratio2+(max_forget_ratio2-min_forget_ratio2).*((1:ps-ps_elite)./(ps-ps_elite))));
        for k=1:ps
            Index2=randperm(D);
            forget_Dim2(k,Index2(1:forget_Num2(k)))=1;
        end
        pmiu2(:,1:D)=repmat(pmiu2(:,1),1,D);
        radius2=max(pos)-min(pos);
        radius2=repmat(radius2,ps,1);
        pmiu2(:,:)=pmiu2(:,:).*radius2*factor;       
        pmiu2=pmiu2.*forget_Dim2;
        
        pbest_stop=zeros(1,ps);
        pbest_improve=zeros(1,ps);
        gbest_stop=0;
    end
    
    gbest_changed=0;  % 全局最优解是否得到更新
    for k=1:ps
        [tmp,tmpid]=min(pbestval(neighbor(k,:)));

        aa(k,:)=c1(k,:).*rand(1,D).*(pbest(k,:)-pos(k,:))+c2(k,:).*rand(1,D).*(pbest(neighbor(k,tmpid),:).*(1+pmiu2(k,:).*randn(1,D))-pos(k,:))...
               +c3(k,:).*rand(1,D).*(gbestrep(k,:).*(1+pmiu(k,:).*randn(1,D))-pos(k,:));
   
        vel(k,:)=iwt(i).*vel(k,:)+aa(k,:); 
        vel(k,:)=(vel(k,:)>mv).*mv+(vel(k,:)<=mv).*vel(k,:); 
        vel(k,:)=(vel(k,:)<(-mv)).*(-mv)+(vel(k,:)>=(-mv)).*vel(k,:);
        pos(k,:)=pos(k,:)+vel(k,:); 

        if rand<0.5
            pos(k,:)=((pos(k,:)<=VRmax(k,:))&(pos(k,:)>=VRmin(k,:))).*pos(k,:)...
                +(1-(pos(k,:)<=VRmax(k,:))&(pos(k,:)>=VRmin(k,:))).*(VRmin(k,:)+(VRmax(k,:)-VRmin(k,:)).*rand(1,D)); 
        else
            pos(k,:)=((pos(k,:)>=VRmin(k,:))&(pos(k,:)<=VRmax(k,:))).*pos(k,:)...
                +(pos(k,:)<VRmin(k,:)).*(VRmin(k,:)+0.1.*(VRmax(k,:)-VRmin(k,:)).*rand(1,D))...
                +(pos(k,:)>VRmax(k,:)).*(VRmax(k,:)-0.1.*(VRmax(k,:)-VRmin(k,:)).*rand(1,D));           
        end
        
%         e(k,1)=feval(fhd,pos(k,:)',varargin{:})-fbias(func_num);
        e(k,1)=f(pos(k,:));
        fitcount=fitcount+1;
       
        if e(k)<=pbestval(k)  % 更新个体的 pbest
            pbest_stop(k)=0;
            pbest(k,:)=pos(k,:);
            pbestval(k)=e(k);
            pbest_improve(k)=pbest_improve(k)+1;
        else
            pbest_stop(k)=pbest_stop(k)+1;
        end
      
        
        if pbestval(k)<gbestval  % 更新种群的 gbest
            gbest=pbest(k,:);
            gbestval=pbestval(k);
            gbestrep=repmat(gbest,ps,1);%update the gbest

             %%%%%%%%%%%%%%
            gbest_changed=1;
        end
       Result(kk)=gbestval;
        %%%% 以下为画图用
        new = fitcount;
        yyy(old:new) = min(gbestval);
        old = new;     
    end
    
    if gbest_changed==1  % 更新 gbest 停滞代数
       gbest_stop=0;
    else
       gbest_stop=gbest_stop+1;
    end
           
    [B_improve IX_improve]=sort(pbest_improve,'descend');
    [B IX]=sort(pbestval);
    selectedid=[]; count=0;

    selectedid(1:n_elite)=IX(1:n_elite);

    good_cmiu1=c1(selectedid,:);
    good_cmiu2=c2(selectedid,:);
    good_cmiu3=c3(selectedid,:);
    kk=kk+1;
end

%%
figure(1)
plot(Result,'r-*','linewidth',1);
title('收敛过程');
axis tight
figure(2)

semilogy(Result,'r-*','linewidth',1);
title('收敛过程');
axis tight
Result';
%%
function [c1, c2, c3] = randFCR(NP, D, cmiu1, low1, up1, cmiu2, low2, up2, cmiu3, low3, up3, delta)

%% generate c1 - c2
% c1 = cmiu1 + delta * randn(NP, D);
[m,n]=size(cmiu1);
c1 = cmiu1 + delta * tan(pi * (rand(m, n) - 0.5));
c1 = min(up1, max(low1, c1));                % truncated to [low1 up1]


% c2 = cmiu2 + delta * randn(NP, D);
[m,n]=size(cmiu2);
c2 = cmiu2 + delta * tan(pi * (rand(m, n) - 0.5));
c2 = min(up2, max(low2, c2));                % truncated to [0 1]

% c3 = cmiu3 + delta * randn(NP, D);
[m,n]=size(cmiu3);
c3 = cmiu3 + delta * tan(pi * (rand(m, n) - 0.5));
c3 = min(up3, max(low3, c3));                % truncated to [0 1]

end

% Cauchy distribution: cauchypdf = @(x, mu, delta) 1/pi*delta./((x-mu).^2+delta^2)
function result = randCauchy(m, n, mu, delta)
% http://en.wikipedia.org/wiki/Cauchy_distribution
result = mu + delta * tan(pi * (rand(m, n) - 0.5));
end
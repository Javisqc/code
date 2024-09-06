% 
% clc; clear all; close all;
% %% Predefine some variables for PSO
% sz = 100;                                                                   % swarm size of pso
% dim = 30;                                                                  % dimension of particle's
% xmax=100;
% xmin=-xmax;
% 
% %  32      5.21       600       10      100     30          10      
% %  Ackley  Rastrigin  Griewank  Alpine  Sphere  Rosenbrock  Schwefel
% X=zeros(sz,dim);                                                           % position of particles
% V=zeros(sz,dim);                                                           % velocity of particles
% 
% Pbest = zeros(sz, dim);                                                    % individual's best position
% fPbest = zeros(sz, 1);
% 
% Gbest = zeros(1, dim);                                                     % global best position
% fGbest = 0;
% 
% fhistory = [ ];                                                            % the best fitness of history
% Xhistory = [ ];                                                            % the best position of history
% 
% fvalue = zeros(sz, 1);
% for k=1:1
%     %% Initialization for pso variables
%     % upper and lower limit of particle's position
%     
%     X = rand(sz, dim);
%     for i = 1 : dim
%         X(:, i)  = repmat(xmin, sz, 1) + (xmax - xmin)*X(:, i);
%     end
%     
%     v_min = -4; v_max = 4;
%     V = v_min + (v_max - v_min)*rand(sz, dim);
%     c1= 2; c2 = 2;
%     w_max = 0.9; w_min = 0.4;
%     iter_max = 5000;                                                          % maximum number of iteration
%     m = 1;
%     %delay=0.02;                                                               %延迟时间
%     %% calculate fitness
%     for s=1:1:sz
%         tempval = f(X(s,:));
%         fvalue(s)=tempval;
%     end
%     %% update global best and individual best
%     [fmin, ind] = min(fvalue);
%     fGbest = fmin;
%     Gbest = X(ind, :);
%     
%     Pbest = X;
%     fPbest = fvalue;
%     
%     fhistory = [fhistory; fGbest];
%     Xhistory = [ Xhistory; [0,Gbest] ];
%     
%     iter = 0;
%     %% pso main loop
%     mask_min = repmat(xmin, 1, dim);
%     mask_max = repmat(xmax, 1, dim);
%     for i = 1:sz
%         PC(i) = 0.05 + 0.45 * (exp(10*(i-1)/(sz-1))-1)/(exp(10)-1);
%     end
%     flag = zeros(sz,1);
%     %figure;
%     %hold on;
%     %plot(X(:,1),X(:,2),'r.');
%     %title('0','fontname','Times New Roman','Color','b','FontSize',16);
%     
%     while iter<iter_max
%         %pause(delay);
%         %cla;
%         
%         %CLPSO
%         for i=1:sz
%             if mod(flag(i),m)==0;
%                 for j= 1:dim
%                     r = rand;
%                     if r < PC(i)
%                         a=randperm(sz);
%                         b=a(1:2);
%                         fchrows = fPbest(b(1)) < fPbest(b(2));
%                         if fchrows == 1
%                             Pbest(i,j) = Pbest(b(1),j);
%                         else
%                             Pbest(i,j) = Pbest(b(2),j);
%                         end
%                     end
%                 end
%                 flag(i)=0;
%             end
%         end
%         
%         w = w_max - iter*(w_max - w_min)/iter_max;
%         V = w*V + c2*rand(sz, dim).*(Pbest - X);
%         % check if V is out range of [v_min v_max]
%         chrows1 = V > v_max;
%         V(find(chrows1)) = v_max;      %[row,col V] = find(X, ...)  查询满足一定条件的元素的行和列
%         chrows2 = V < v_min;
%         V(find(chrows2)) = v_min;
%         
%         X = X +V;
%         
%         for s=1:1:sz
%             tempval = f(X(s,:));
%             fvalue(s)=tempval;
%         end
%         
%         % check if X is out range of xrange
%         for i = 1:sz
%             min_throw = X(i,:) < mask_min;
%             max_throw = X(i,:) > mask_max;
%             mi = sum(sum(min_throw.*min_throw));
%             ma = sum(sum(max_throw.*min_throw));
%             mm = mi + ma;
%             if mm==0
%                 if fvalue(i) < fPbest(i)
%                     fPbest(i) = fvalue(i);
%                     Pbest(i,:) = X(i,:);
%                 else
%                     flag(i) = flag(i) + 1;
%                 end
%             end
%         end
%         % check if X is out range of xrange
%         %min_throw = X <= mask_min;
%         %min_keep = X > mask_min;
%         %max_throw = X >= mask_max;
%         %max_keep = X < mask_max;
%         %X = ( min_throw.*mask_min ) + ( min_keep.*X );
%         %X = ( max_throw.*mask_max ) + ( max_keep.*X );
%         
%         % update individual's best
%         %chrows = f < fPbest;
%         %fPbest(find(chrows)) = f(find(chrows));
%         %Pbest(find(chrows), :) = X(find(chrows), :);
%         
%         % update global best
%         [fmin, ind] = min(fPbest);
%         if fmin < fGbest
%             fGbest = fmin;
%             Gbest = Pbest(ind, :);
%         end
%         
%         
%         %plot(X(:,1),X(:,2),'r.');
%         %title(iter,'fontname','Times New Roman','Color','b','FontSize',16);
%         
%         fhistory = [fhistory; fGbest];
%         Xhistory = [ Xhistory; [iter,Gbest]];
%         
%         iter = iter +1;
%     end
%     % End of loop
%     fhistory1 = fhistory;
%     fGbest1 = fGbest;
%     %% output the final result
%     figure(1)
%     plot(fhistory);
%     title('最优个体适应度','fontsize',12);
%     xlabel('进化代数','fontsize',12);
%     ylabel('适应度','fontsize',12);
%     figure (2)
%     semilogy(fhistory,'r-*','linewidth',2);
%     time(k)=fhistory(end);
% end




%%  
        %modified by Tang 2007-05-14 at Tongji University

% function [gbest_t,gbestval_t,fitcount]= CLPSO(fhd,Max_Gen,Max_FES,Particle_Number,Dimension,VRmin,VRmax)
%[gbest,gbestval,fitcount]= CLPSO_new_func('f8',3500,200000,30,30,-5.12,5.12)


Max_Gen=5000
Max_FES=2000000
Particle_Number=100;
Dimension=30;
VRmax=500;
VRmin=-VRmax;

rand('state',sum(100*clock));

%=============================================================
me=Max_Gen;
ps=Particle_Number;%population size
D=Dimension;
t=0:1/(ps-1):1;t=5.*t;
%Pc=0.0+(0.5-0.0).*(exp(t)-exp(t(1)))./(exp(t(ps))-exp(t(1)));  %Learning proportion Pc , which determines how many dimensions are chosen to learn from other particles' pbests.
Pc=0.05+0.45.*(exp(t)-exp(t(1)))./(exp(t(ps))-exp(t(1)));  
% Pc=0.5.*ones(1,ps);
dm=3*ones(ps,1); %m dimensions are randomly chosen to learn from the gbest. Some of the remaining D-m dimensions are randomly chosen to learn from some randomly chosen particles' pbests and the remaining dimensions learn from its pbest
iwt=0.9-(1:me)*(0.7/me);%inertia weight
% iwt=0.729-(1:me)*(0.0/me);
cc=[1.49445 1.49445]; %acceleration constants

if length(VRmin)==1
    VRmin=repmat(VRmin,1,D);
    VRmax=repmat(VRmax,1,D);
end
mv=0.2*(VRmax-VRmin);%max velocity
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);
Vmin=repmat(-mv,ps,1);
Vmax=-Vmin;
pos=VRmin+(VRmax-VRmin).*rand(ps,D);    %initialize the position value

for i=1:ps;
    e(i,1)=f(pos(i,:));    %initialize the fitness value
end

fitcount=ps;
vel=Vmin+2.*Vmax.*rand(ps,D);     %initialize the velocity of the particles
pbest=pos;     %initialize the pbest (psXD)
pbestval=e;    %initialize the pbest's fitness value
[gbestval,gbestid]=min(pbestval);    %initialize the gbest's fitness value
gbestval_t(1)=gbestval;    %initialize the pbest's fitness value
gbest=pbest(gbestid,:);     %initialize the gbest value (1XD)
gbestrep=repmat(gbest,ps,1);
gbest_t(1,:)=gbest;     %
stay_num=zeros(ps,1); 

    ai=zeros(ps,D);
    f_pbest=1:ps;f_pbest=repmat(f_pbest',1,D);
    for k=1:ps
        ar=randperm(D);%??????????
        ai(k,ar(1:dm(k)))=1;
        fi1=ceil(ps*rand(1,D));
        fi2=ceil(ps*rand(1,D));
        fi=(pbestval(fi1)<pbestval(fi2))'.*fi1+(pbestval(fi1)>=pbestval(fi2))'.*fi2;
        bi=ceil(rand(1,D)-1+Pc(k));
        if bi==zeros(1,D)
            rc=randperm(D);
            bi(rc(1))=1;
        end
        f_pbest(k,:)=bi.*fi+(1-bi).*f_pbest(k,:);
    end
    
    stop_num=0;
    i=1;

 while i<me&&fitcount<Max_FES
     i=i+1;
     for k=1:ps;
         if stay_num(k)>=5  %????????????
   %     if round(i/10)==i/10%|stay_num(k)>=5
             stay_num(k)=0;
             ai(k,:)=zeros(1,D);
             f_pbest(k,:)=k.*ones(1,D);
             ar=randperm(D);
             ai(k,ar(1:dm(k)))=1;
             fi1=ceil(ps*rand(1,D));
             fi2=ceil(ps*rand(1,D));
             fi=(pbestval(fi1)<pbestval(fi2))'.*fi1+(pbestval(fi1)>=pbestval(fi2))'.*fi2;
             bi=ceil(rand(1,D)-1+Pc(k));
             if bi==zeros(1,D)
                rc=randperm(D);
                bi(rc(1))=1;
             end
             f_pbest(k,:)=bi.*fi+(1-bi).*f_pbest(k,:);
         end
         for dimcnt=1:D
             pbest_f(k,dimcnt)=pbest(f_pbest(k,dimcnt),dimcnt);
         end
         aa(k,:)=cc(1).*(1-ai(k,:)).*rand(1,D).*(pbest_f(k,:)-pos(k,:))+cc(2).*ai(k,:).*rand(1,D).*(gbestrep(k,:)-pos(k,:));%~~~~~~~~~~~~~~~~~~~~~~  
         vel(k,:)=iwt(i).*vel(k,:)+aa(k,:);
         vel(k,:)=(vel(k,:)>mv).*mv+(vel(k,:)<=mv).*vel(k,:);
         vel(k,:)=(vel(k,:)<(-mv)).*(-mv)+(vel(k,:)>=(-mv)).*vel(k,:);
         pos(k,:)=pos(k,:)+vel(k,:);% update the position
         if (sum(pos(k,:)>VRmax(k,:))+sum(pos(k,:)<VRmin(k,:)))==0;
             e(k,1)=f(pos(k,:));%%%%%%%%%%%%%%%%
             fitcount=fitcount+1;
             tmp=(pbestval(k)<=e(k));
             if tmp==1
                 stay_num(k)=stay_num(k)+1;
             end
             temp=repmat(tmp,1,D);
             pbest(k,:)=temp.*pbest(k,:)+(1-temp).*pos(k,:);
             pbestval(k)=tmp.*pbestval(k)+(1-tmp).*e(k);%update the pbest
             if pbestval(k)<gbestval
                 gbest=pbest(k,:);
                 gbestval=pbestval(k);
                 gbestrep=repmat(gbest,ps,1);%update the gbest
             end
         end         
     end
%=============Plot the position=================     
%{
      if round(i/100)==i/100;
      plot(pos(:,D-1),pos(:,D),'b*');hold on;
      for k=1:floor(D/2)
          plot(gbest(:,2*k-1),gbest(:,2*k),'r*');%%%%%%%%%%%%%%%%
      end
      hold off
      title(['PSO: ',num2str(i),' generations, Gbestval=',num2str(gbestval)]);  
      axis([VRmin(1,D-1),VRmax(1,D-1),VRmin(1,D),VRmax(1,D)])
      drawnow
end
%}
%=============end Plot=================     
%{
     if fitcount>=Max_FES
         break;
     end
     if (i==me)&&(fitcount<Max_FES)
        i=i-1;
end
%}
     gbest_t(i,:)=gbest;    %
     gbestval_t(i)=gbestval;   %
     fprintf(1,'Iteration: %d,  Best: %f\n',i,gbestval);
     fprintf(1,'BestX: %f\n',gbest);
 end

 %=====================plot the results============
for ii=1:D
    hold on;
    plot(gbest_t(:,ii));
end
hold off;

figure(1)
semilogy(gbestval_t,'r-*','linewidth',2);
title('收敛过程');
axis tight
figure(2)
plot(gbestval_t,'r-*','linewidth',2);
title('收敛过程');
axis tight
gbestval_t'
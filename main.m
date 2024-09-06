clear all
clc

global fbias

func_num=1;
D=30;
Xmin=-100;
Xmax=100;


pop_size=60;%2^(4+floor(log2(sqrt(D))));
fes_max=10000*D;
iter_max=ceil(fes_max/pop_size);

runtimes=1;
fhd=f;


% CEC2013
fbias=[-1400, -1300, -1200, -1100, -1000,...
          -900,  -800,  -700,  -600,  -500,...
          -400,  -300,  -200,  -100,   100,...
          200,    300,   400,   500,   600,...
          700,    800,   900,  1000,  1100,...
          1200,   1300,  1400 ];
      
AccErr=[1e-6, 100, 1e7, 100, 1e-6];
t=ones(1,23)*100;
AccErr=[AccErr t];
% jingdu=1e-8;

% fun=[4,11,12,14,15,17,19,22,23];%[1,5,8,20,26,28];
fun=[1:1];
for func_index=1:length(fun)
    func_num=fun(func_index);
    suc_times = 0; 
    timeusage=0;
    fesusage=0;
    count=0; 
    jingdu = 0;%AccErr(func_num);
    for runs=1:runtimes
        suc = 0;
        suc_fes = 0;   % added by us
        
        tic;
      

        [gbest,gbestval,FES,suc,suc_fes]= XPSO(jingdu,func_num,fhd,D,pop_size,iter_max,fes_max,Xmin,Xmax,func_num);

        t=toc;
        time_usage(runs,func_index)=t;
%         gbestval=gbestval-fbias(func_num);
        xbest(runs,:)=gbest(1,:);
        fbest(runs,func_num)=gbestval;
        fprintf('The result of the %d runs is��%1.4e\n',runs,gbestval);
        suc_times = suc_times + suc;
        if suc == 1  % ���ﵽ�趨����ʱ��ͳ�����ʱ           
            fesusage = fesusage + suc_fes;   % �ﵽ����ʱ�� fes            
        end    
        
    end
    
    % �����������EXCEL�ļ�
%     sheet = 1;   k = 2; 
%     for alg = 1:5
%         d = k + alg*runtimes;
%         row = num2str(d);
%         col = setstr(func_num-1+'B');
% %         col = char(c);
%         xlRange = strcat(col,row);
%         xlswrite(filename,fitbest(:,alg),sheet,xlRange);
%         k = k + 9;
%     end
    %%%%%%%%%
    
    
     %% ����Ϊ�����������趨��⾫�������µĳɹ��ʡ��������۴����Լ�����ʱ��
    SR(1,func_num) = suc_times/runtimes;
    if suc_times>0
        FEs(1,func_num) = fesusage/suc_times;  % ���㾫�ȵĶ�����������ĵ�ƽ�� fes��δ���ǲ����㾫�ȵ�����
        SP (1,func_num) = fes_max*(1-SR(1,func_num))/SR(1,func_num) + FEs(1,func_num); % �ۺ��������㷨�����ܣ��ȿ��ǳɹ��ģ�Ҳ����δ�ɹ���
    else
        FEs(1,func_num) = -1;  % ���㾫�ȵĶ�����������ĵ�ƽ�� fes��δ���ǲ����㾫�ȵ�����
        SP (1,func_num) = -1; % �ۺ��������㷨�����ܣ��ȿ��ǳɹ��ģ�Ҳ����δ�ɹ���
    end
    f_mean(func_num)=mean(fbest(:,func_num));
    fprintf('\nFunction F%d :\nAvg. fitness = %1.2e(%1.2e)\n\n',func_num,mean(fbest(:,func_num)),std(fbest(:,func_num)));    
    fprintf(' -------------------------------------------------- \n');
end




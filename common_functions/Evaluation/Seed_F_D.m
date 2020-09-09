n_stacks = 6;
[S,f1s1] = fun_min_d('user_user',n_stacks);
[S,f1s2] = fun_min_d('net03+Astar03+D04_ImfImo_15_1_200_CutNet98_onlyMST',n_stacks);
% n_stacks = 9;
% [S,f1s1] = fun_min_d('OP_user_user_Amazon',n_stacks);
% [S,f1s2] = fun_min_d('OP_Astar05+D05_ImfImo_15_1',n_stacks);

figure,plot(S,f1s1,'*-','color','k')
hold on
plot(S,f1s2,'*-','color','b')
ylim([0,1])
axis square

function [S,f1s] = fun_min_d(foldername,n_stacks)

f1s = zeros(8,1);
S = 0.5:0.5:4.0;
for s = 1:8
    f1 = zeros(n_stacks,1);
    for z = 1:n_stacks
        TP = 0;
        FP = 0;
        FN = 0;
        load(['E:\Shih-Luen\Lab\Publications\Thesis\Thesis\emf\Roahn_Metrics\',foldername,'\',num2str(z),'_AS.mat'])
        for i = 1:length(r_G)
            min_d = cal_min_d(r_G(i,:),r_A);
            if min_d <= S(s)
                TP = TP + 1;
            else
                FN = FN + 1;
            end
        end
        
        for i = 1:length(r_A)
            min_d = cal_min_d(r_A(i,:),r_G);
            if min_d <= S(s)
                TP = TP + 1;
            else
                FP = FP + 1;
            end
        end
        
        f1(z) = TP/(TP+(FP+FN)/2);
    end
    f1s(s) = mean(f1);
end
end

function [min_d] = cal_min_d(p,r)
min_d = min(sqrt(sum((r - p).^2,2)));
end
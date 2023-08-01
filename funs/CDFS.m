function [U, f] = CDFS(X, Y, k)
    % Coordinate Descent Feature Selction CDFS
    % X: labeled data, n by d
    % Y: label matrix, n by c
    % k: the number of selected features
    % 
    % Written by Han Zhang 2022/1/15
    %
    % for i = 1:size(X, 1)
    %    X(i,:)=X(i,:)./norm(X(i,:),'fro');
    % end
    
    %addpath('.\utils');
    n = size(X, 1);
    H = eye(n) - ones(n, 1) * ones(1, n) * 1 / n;
    St = X' * H * X;
    Q = inv(Y' * Y);
    Sb = X' * H * Y * Q * Y' * H * X;
    d = size(Sb, 1);
    [~, idx] = sort(diag(Sb), 'descend');
    init_f = idx(1 : k);
    init_U = zeros(d , k);
    for i = 1 : k
        init_U(init_f(i), i) = 1;
    end
    U = init_U;
    f = init_f;
    us = eye(d); 
    t1 = clock; 
    obj_old = trace((U' * Sb * U) * (U' * St * U) ^ -1);
    t2 = clock;
    t1 = clock;
    for i = 1 : k
        idx = 1 : k;
        idx(i) = [];
        Ui = U(:, idx);
        Db = Ui' * Sb * Ui;
        invDt = (Ui' * St * Ui) ^ -1; 
        for j = 1: d
            temp_U = U;
            if ~ismember(j, f)
                temp_U(:, i) = us(:, j);
                ui = temp_U(:, i);
                obj = compute_obj(ui, Ui, Db, invDt, Sb, St); 
                err = obj - obj_old;
                if err > 0
                    f(i) = j;
                    obj_old = obj;
                    U(:, i) = us(:, j);
                end
            end      
        end
    end
    t2 = clock;
end


function [obj] = compute_obj(ui, Ui, Db, invDt, Sb, St)
    Ab = ui' * Sb * ui;
    Bb = ui' * Sb * Ui;
    USbU = [Ab, Bb; Bb', Db];
    At = ui' * St * ui;
    Bt = ui' * St * Ui;
    invPA = (At - Bt * invDt * Bt') ^- 1;
    invPB = -invPA * Bt * invDt;
    invPD = invDt + invDt * Bt' * (-invPB);
    invUStU = [invPA, invPB; invPB', invPD];
    obj = trace(USbU * invUStU);
end
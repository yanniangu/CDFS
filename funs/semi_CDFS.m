function [Yu, U, f] = semi_CDFS(Xl, YL, Xu, opt)
    % semi-supervised coordinate descent feature selection
    % Xl: labeled data :nl by d
    % YL: label matrix, nl by c
    % Xu: Unlabeled data, nu by d
    % opt:  -k: the number of selected features
    %       -c: the real class number
    %       -r1: regualrization parameter
    %       -MaxIter: # of iterations
    %       -initW: 1 represents a good initilization for W; 0 represents a
    %       random inilization.
    %       -rho & mu: the parameters of ALM algorithm
    % Written by Yannian Gu.
    
    [d, nu] = size(Xu);
    X = [Xl, Xu];

    %% ----------Remove Centers---------- %%
    n = size(X, 2);
    X0 = X;
    mX0 = mean(X0, 2);
    X1 = X0 - mX0 * ones(1, n);
    scal = 1 ./ sqrt(sum(X1 .* X1) + eps);
    scalMat = sparse(diag(scal));
    X = X1 * scalMat;
    Xl = X(:, 1:n-nu);
    Xu = X(:, n-nu+1:end);
    T = eye(opt.c);%label encoding matrix 

    %% ----------Initialization---------- %%
    Yu = zeros(nu, opt.c);
    Y = zeros(n, opt.c);
    Y(1:n-nu, :) = YL;
    XX = Xl * Xl';
    XY = Xl * YL;
    b = mean(YL - Xl' * ((XX + 0.1 * eye(d)) \ XY));
    if opt.initW == 1
        [~, W, ~, ~] = FSRobust_ALM(X, F, opt.k, opt.mu, opt.rho, 50);
    elseif opt.initW == 2
        W = initializationW(Xl, YL, opt.k);
    else
        W = rand(d,opt.c);
    end
    
    %% ----------Main Code---------- %%
    obj = zeros(opt.MaxIter, 1);
    for iter = 1: opt.MaxIter
    
        %fix W, update Fu
        pred = Xu' * W + repmat(b, nu, 1);
        for j = 1:opt.c
            v = pred - repmat(T(j, :), nu, 1);
            p(:, j) = sum(v.*v,2);%p(i,k)
        end
        if opt.r1 == 1
            Yu = zeros(nu, opt.c);
            [~, idx] = min(p, [], 2);
            for i = 1:nu
                Yu(i,idx(i)) = 1;
            end
        else 
            mm = (opt.r1 .* p) .^ (1 / (1 - opt.r1 - eps));
            s0 = sum(mm, 2);%length(find(isnan(s0)))
            Yu = mm ./ repmat(s0, 1, opt.c);%sum(Yu,2)
        end
        Y(n - nu + 1: end, :) = Yu;
    
        % fix Fu, update U
        G = Y .^ opt.r1;
        S = diag(sum(G, 2));
        St = X * S * X';
        Sb = X * G * G' * X';
        sd = size(Sb, 1);
        [~,idx] = sort(diag(Sb),'descend');
        init_f = idx(1:opt.k);
        init_U = zeros(sd,opt.k);
        for i = 1:opt.k
            init_U(init_f(i),i) = 1;
        end 
        U = init_U;   
        f = init_f;
        us = eye(sd); 
        t1 = clock; 
        obj_old = trace((U'*Sb*U)*(U'*St*U)^-1);
        t2 = clock;
        t1 = clock;
        for i = 1:opt.k
            idx = 1:opt.k;
            idx(i) = [];
            Ui = U(:, idx);
            Db = Ui'*Sb*Ui;
            invDt = (Ui'*St*Ui)^-1; 
            for j = 1:sd
                temp_U = U;
                if ~ismember(j,f)
                    temp_U(:,i) = us(:,j);
                    ui = temp_U(:,i);
                    obj = compute_obj(ui, Ui, Db, invDt, Sb, St); 
                    err = obj-obj_old;
                    if err>0
                        f(i) = j;
                        obj_old = obj;
                        U(:,i) = us(:,j);
                    end
                end      
            end
        end
        
        % compute W
        V = inv(U' * X * S * X' * U) * U' * X * G;
        W = U * V;
        t2 = clock;
    end
end

function [ obj ] = compute_obj(ui, Ui, Db, invDt, Sb, St)
    Ab = ui'*Sb*ui;
    Bb = ui'*Sb*Ui;
    USbU = [Ab, Bb; Bb', Db];
    At = ui'*St*ui;
    Bt = ui'*St*Ui;
    invPA = (At-Bt*invDt*Bt')^-1;
    invPB = -invPA*Bt*invDt;
    invPD = invDt+invDt*Bt'*(-invPB);
    invUStU = [invPA, invPB; invPB', invPD];
    obj = trace(USbU*invUStU);
end

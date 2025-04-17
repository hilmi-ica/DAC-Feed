function tree = DistTree(dval_list, Nr)
    Nd = numel(dval_list);
    
    % Generate all combinations for one time step
    [G{1:Nd}] = ndgrid(dval_list{:});
    one_step = reshape(cat(Nd+1, G{:}), [], Nd);
    Nv = size(one_step, 1);

    % Total scenarios
    total_scenarios = Nv^Nr;
    
    % Generate all index combinations for Nr steps
    idx_cells = cell(1, Nr);
    [idx_cells{:}] = ndgrid(1:Nv);
    idx_combo = reshape(cat(Nr+1, idx_cells{:}), [], Nr);

    % Build the tree
    tree = zeros(total_scenarios, Nd, Nr);
    for t = 1:Nr
        tree(:,:,t) = one_step(idx_combo(:,t), :);
    end
end
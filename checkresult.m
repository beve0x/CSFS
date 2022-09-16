function result = checkresult(pred_label, test_label, cost_setting)

total_cost = 0;
err_OI = 0;
err_IO = 0;
err_II = 0;

test_num = length(test_label);
test_num_pos=length(find(test_label==1));
test_num_neg=length(find(test_label==-1));


for i = 1 : test_num
    if pred_label(i) ~= test_label(i)
        if test_label(i) == 1 && pred_label(i) == -1
            total_cost = total_cost + cost_setting.C_OI;
            err_OI = err_OI + 1;
            %fprintf('id of errig %d ',i);
        end
        if test_label(i) == -1 && pred_label(i) == 1
            total_cost = total_cost + cost_setting.C_IO;
            err_IO = err_IO + 1;
            %fprintf('id of errgi %d ',i);
        end        
    end
end
total_err = err_OI + err_IO;

result.total_cost = total_cost;
result.total_err = total_err/test_num;
result.err_OI = err_OI/test_num_pos;
result.err_IO = err_IO/test_num_neg;
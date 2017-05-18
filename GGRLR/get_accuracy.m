function accuracy = get_accuracy(s_test,l_test,x)
t_predict   = l_test.*(s_test'*x);
num_correct = sum(t_predict>0);
accuracy    = num_correct/length(l_test);
end
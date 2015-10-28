[tr_labels,tr_feats,te_labels,te_feats] = read_data(ntrain,data_dir);

fprintf('\n\n --- LINEAR SVM --- \n');
config.KERNEL_TYPE = 0;    % linear SVM
models = train_models(tr_labels,tr_feats);
[acc_linear, pl_linear] = predict_labels(models,te_labels,te_feats);


fprintf('\n\n --- INTERSECTION KERNEL SVM --- \n');
config.KERNEL_TYPE = 4;
models = train_models(tr_labels,tr_feats);
[acc_iksvm, pl_iksvm] = predict_labels(models,te_labels,te_feats);

fprintf('\n\n --- POLYNOMIAL KERNEL SVM --- \n');
config.KERNEL_TYPE = 1;
models = train_models(tr_labels,tr_feats);
[acc_poly, pl_poly] = predict_labels(models,te_labels,te_feats);

fprintf('\n\n --- RBF KERNEL SVM --- \n');
config.KERNEL_TYPE = 2;
models = train_models(tr_labels,tr_feats);
[acc_rbf, pl_rbf] = predict_labels(models,te_labels,te_feats);

fprintf('\t-------------------------\n');
fprintf('\t Method\t Acc(%%)\t Err(%%)\n');
fprintf('\t-------------------------\n');
fprintf('\t LINEAR\t%.2f%%\t %.2f%%\n',acc_linear,100-acc_linear);
fprintf('\t IKSVM\t%.2f%%\t %.2f%%\n',acc_iksvm,100-acc_iksvm);
fprintf('\t POLY\t%.2f%%\t %.2f%%\n',acc_poly,100-acc_poly);
fprintf('\t RBF\t%.2f%%\t %.2f%%\n',acc_rbf,100-acc_rbf);

figure;
bar([acc_linear; acc_iksvm; acc_poly; acc_rbf]);
set(gca,'XTickLabel',{'Linear SVM','IKSVM','POLY SVM','RBF SVM'});
title(sprintf('MNIST dataset performance (ntrain %i, raw)',ntrain));
ylabel('Accuracy(%)'); colormap summer; grid on;

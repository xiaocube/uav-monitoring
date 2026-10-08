% 精确率-召回率曲线绘制脚本
% 运行方式：在 Matlab 中 cd 到项目根目录，运行此脚本
% 输入数据：各模型 results.csv 文件中的 precision 和 recall 数据
% 输出：P-R 曲线图

clear; clc; close all;

model_names = {
    'drone-v1 (YOLOv8n)', ...
    'drone-v1-2 (YOLOv8n)', ...
    'drone-v2-dronetrack', ...
    'drone-v2-final', ...
    'skyguard-v2-uav-3 (YOLOv11s)'
};

model_paths = {
    'archive/成功率不高-旧训练数据/训练结果/drone-v1/results.csv', ...
    'archive/成功率不高-旧训练数据/训练结果/drone-v1-2/results.csv', ...
    'archive/成功率不高-旧训练数据/训练结果/drone-v2-dronetrack/results.csv', ...
    'archive/成功率不高-旧训练数据/训练结果/runs/models/trained/drone-v2-dronetrack-final/results.csv', ...
    'runs/detect/models/trained/skyguard-v2-uav-3/results.csv'
};

colors = [
    0.8500 0.3250 0.0980; ...  % red
    0.9290 0.6940 0.1250; ...  % orange
    0.4940 0.1840 0.5560; ...  % purple
    0.4660 0.6740 0.1880; ...  % green
    0.0000 0.4470 0.7410; ...  % blue
];

figure('Position', [100, 100, 1000, 700]);

for idx = 1:length(model_names)
    csv_data = readtable(model_paths{idx});
    precision = csv_data.metrics_precision_B_;
    recall = csv_data.metrics_recall_B_;
    
    [recall_sorted, sort_idx] = sort(recall);
    precision_sorted = precision(sort_idx);
    
    plot(recall_sorted, precision_sorted, '-', 'Color', colors(idx, :), 'LineWidth', 2, ...
         'Marker', 'o', 'MarkerSize', 5, 'DisplayName', model_names{idx});
    hold on;
end

title('各模型精确率-召回率 (P-R) 曲线对比', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('召回率 (Recall)', 'FontSize', 12);
ylabel('精确率 (Precision)', 'FontSize', 12);
xlim([0.5, 1.0]);
ylim([0.5, 1.0]);
grid on;
legend('Location', 'lower left', 'FontSize', 10);

fprintf('\n各模型最佳指标汇总：\n');
fprintf('========================================\n');
fprintf('%25s | %8s | %8s | %8s\n', '模型', 'mAP50', 'Precision', 'Recall');
fprintf('========================================\n');

for idx = 1:length(model_names)
    csv_data = readtable(model_paths{idx});
    best_map50 = max(csv_data.metrics_mAP50_B_);
    best_p = max(csv_data.metrics_precision_B_);
    best_r = max(csv_data.metrics_recall_B_);
    fprintf('%25s | %8.4f | %8.4f | %8.4f\n', model_names{idx}, best_map50, best_p, best_r);
end

print('paper/charts/pr_curves.png', '-dpng', '-r300');
disp('P-R 曲线图已保存到 paper/charts/pr_curves.png');

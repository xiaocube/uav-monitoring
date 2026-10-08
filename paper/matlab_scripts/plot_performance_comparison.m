% 多模型性能对比柱状图绘制脚本
% 运行方式：在 Matlab 中 cd 到项目根目录，运行此脚本
% 输入数据：各模型 results.csv 文件中的最佳 mAP 指标
% 输出：多模型 mAP/FPS 性能对比柱状图

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

% 提取指标
mAP50 = zeros(1, length(model_names));
mAP50_95 = zeros(1, length(model_names));
precision = zeros(1, length(model_names));
recall = zeros(1, length(model_names));

for idx = 1:length(model_names)
    csv_data = readtable(model_paths{idx});
    mAP50(idx) = max(csv_data.metrics_mAP50_B_);
    mAP50_95(idx) = max(csv_data.metrics_mAP50_95_B_);
    precision(idx) = max(csv_data.metrics_precision_B_);
    recall(idx) = max(csv_data.metrics_recall_B_);
end

% 绘制 mAP50 对比
figure('Position', [100, 100, 1400, 600]);

subplot(1, 2, 1);
bar(mAP50, 'FaceColor', 'flat');
set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, 'FontSize', 9, 'Rotation', 30);
for idx = 1:length(model_names)
    patch(get(bar(idx), 'XData'), get(bar(idx), 'YData'), colors(idx, :), 'FaceColor', colors(idx, :));
end
title('各模型 mAP50 指标对比', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('mAP50', 'FontSize', 12);
ylim([0.8, 1.0]);
grid on;

% 绘制 mAP50-95 对比
subplot(1, 2, 2);
bar(mAP50_95, 'FaceColor', 'flat');
set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, 'FontSize', 9, 'Rotation', 30);
for idx = 1:length(model_names)
    patch(get(bar(idx), 'XData'), get(bar(idx), 'YData'), colors(idx, :), 'FaceColor', colors(idx, :));
end
title('各模型 mAP50-95 指标对比', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('mAP50-95', 'FontSize', 12);
ylim([0.3, 0.7]);
grid on;

sgtitle('多模型检测精度性能对比', 'FontSize', 16, 'FontWeight', 'bold');

print('paper/charts/map_comparison.png', '-dpng', '-r300');
disp('mAP 对比图已保存到 paper/charts/map_comparison.png');

% 绘制 Precision-Recall 散点图
figure('Position', [100, 100, 800, 600]);

scatter(recall, precision, 150, colors, 'filled', 'Marker', 'o');
hold on;
for idx = 1:length(model_names)
    text(recall(idx) + 0.005, precision(idx) + 0.005, model_names{idx}, 'FontSize', 9);
end

title('各模型 Precision-Recall 散点图', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('召回率 (Recall)', 'FontSize', 12);
ylabel('精确率 (Precision)', 'FontSize', 12);
xlim([0.85, 0.96]);
ylim([0.88, 0.99]);
grid on;

print('paper/charts/pr_scatter.png', '-dpng', '-r300');
disp('P-R 散点图已保存到 paper/charts/pr_scatter.png');

% 性能数据汇总
fprintf('\n多模型性能指标汇总：\n');
fprintf('=================================================================\n');
fprintf('%25s | %8s | %10s | %8s | %8s\n', '模型', 'mAP50', 'mAP50-95', 'Precision', 'Recall');
fprintf('=================================================================\n');
for idx = 1:length(model_names)
    fprintf('%25s | %8.4f | %10.4f | %8.4f | %8.4f\n', ...
        model_names{idx}, mAP50(idx), mAP50_95(idx), precision(idx), recall(idx));
end

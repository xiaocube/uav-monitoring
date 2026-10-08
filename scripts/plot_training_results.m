%% YOLOv11 无人机检测模型训练结果可视化
% SkyGuard AI - skyguard-v2-uav-3
% 生成日期: 2026-07-23

clc; clear; close all;

%% 读取数据
data = readmatrix('runs/detect/models/trained/skyguard-v2-uav-3/results.csv', 'NumHeaderLines', 1);

epoch       = data(:, 1);
train_box   = data(:, 3);
train_cls   = data(:, 4);
train_dfl   = data(:, 5);
precision   = data(:, 6);
recall      = data(:, 7);
mAP50       = data(:, 8);
mAP50_95    = data(:, 9);
val_box     = data(:, 10);
val_cls     = data(:, 11);
val_dfl     = data(:, 12);

%% 设置全局样式
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultAxesFontName', 'Helvetica');
set(0, 'DefaultAxesLineWidth', 1.2);
set(0, 'DefaultLineLineWidth', 2);

colors = lines(6);

%% 图1: 训练损失曲线 (3子图)
fig1 = figure('Name', '训练损失曲线', 'Position', [100 100 900 350]);

% Box Loss
subplot(1,3,1);
plot(epoch, train_box, '-', 'Color', colors(1,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('Box Loss');
grid on; set(gca, 'GridAlpha', 0.3);

% Cls Loss
subplot(1,3,2);
plot(epoch, train_cls, '-', 'Color', colors(2,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('Cls Loss');
grid on; set(gca, 'GridAlpha', 0.3);

% DFL Loss
subplot(1,3,3);
plot(epoch, train_dfl, '-', 'Color', colors(3,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('DFL Loss');
grid on; set(gca, 'GridAlpha', 0.3);

sgtitle('训练损失曲线 (Training Loss)', 'FontSize', 14, 'FontWeight', 'bold');

%% 图2: 验证指标曲线
fig2 = figure('Name', '验证指标', 'Position', [100 500 900 400]);

% Precision & Recall
subplot(2,1,1);
plot(epoch, precision, '-o', 'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 4);
hold on;
plot(epoch, recall, '-s', 'Color', colors(2,:), 'LineWidth', 2, 'MarkerSize', 4);
xlabel('Epoch'); ylabel('Score');
title('Precision & Recall');
legend('Precision', 'Recall', 'Location', 'southeast');
grid on; set(gca, 'GridAlpha', 0.3);
ylim([0.7 1.0]);

% mAP50 & mAP50-95
subplot(2,1,2);
plot(epoch, mAP50, '-o', 'Color', colors(5,:), 'LineWidth', 2, 'MarkerSize', 4);
hold on;
plot(epoch, mAP50_95, '-s', 'Color', colors(6,:), 'LineWidth', 2, 'MarkerSize', 4);
xlabel('Epoch'); ylabel('Score');
title('mAP50 & mAP50-95');
legend('mAP50', 'mAP50-95', 'Location', 'southeast');
grid on; set(gca, 'GridAlpha', 0.3);
ylim([0.3 1.0]);

sgtitle('验证指标曲线 (Validation Metrics)', 'FontSize', 14, 'FontWeight', 'bold');

%% 图3: 综合仪表板 (2x2)
fig3 = figure('Name', '训练综合仪表板', 'Position', [200 200 1000 600]);

% 1. 所有训练Loss
subplot(2,2,1);
plot(epoch, train_box, '-', 'Color', colors(1,:), 'LineWidth', 2); hold on;
plot(epoch, train_cls, '-', 'Color', colors(2,:), 'LineWidth', 2);
plot(epoch, train_dfl, '-', 'Color', colors(3,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('训练损失 (Training Loss)');
legend('Box', 'Cls', 'DFL', 'Location', 'northeast');
grid on; set(gca, 'GridAlpha', 0.3);

% 2. 验证Loss
subplot(2,2,2);
plot(epoch, val_box, '-', 'Color', colors(1,:), 'LineWidth', 2); hold on;
plot(epoch, val_cls, '-', 'Color', colors(2,:), 'LineWidth', 2);
plot(epoch, val_dfl, '-', 'Color', colors(3,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('验证损失 (Val Loss)');
legend('Box', 'Cls', 'DFL', 'Location', 'northeast');
grid on; set(gca, 'GridAlpha', 0.3);

% 3. Precision & Recall
subplot(2,2,3);
area_x = [epoch, fliplr(epoch)];
area_y = [precision, fliplr(recall)];
fill([epoch', fliplr(epoch')], [precision', fliplr(recall')'], ...
    colors(1,:), 'FaceAlpha', 0.1, 'EdgeColor', 'none'); hold on;
plot(epoch, precision, '-o', 'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 3);
plot(epoch, recall, '-s', 'Color', colors(2,:), 'LineWidth', 2, 'MarkerSize', 3);
xlabel('Epoch'); ylabel('Score');
title('Precision & Recall');
legend('Precision', 'Recall', 'Location', 'southeast');
grid on; set(gca, 'GridAlpha', 0.3);
ylim([0.7 1.0]);

% 4. mAP 指标 + 最佳点标注
subplot(2,2,4);
plot(epoch, mAP50, '-o', 'Color', colors(5,:), 'LineWidth', 2, 'MarkerSize', 4); hold on;
plot(epoch, mAP50_95, '-s', 'Color', colors(6,:), 'LineWidth', 2, 'MarkerSize', 4);

% 标注最佳mAP50点
[best_mAP50, best_idx] = max(mAP50);
plot(epoch(best_idx), best_mAP50, 'p', 'MarkerSize', 15, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
text(epoch(best_idx)+0.5, best_mAP50-0.03, ...
    sprintf('Best: %.3f (Epoch %d)', best_mAP50, epoch(best_idx)), ...
    'FontSize', 9, 'Color', 'r');

% 标注最佳mAP50-95点
[best_mAP95, best_idx95] = max(mAP50_95);
plot(epoch(best_idx95), best_mAP95, 'p', 'MarkerSize', 15, ...
    'MarkerFaceColor', [0.8 0.4 0], 'MarkerEdgeColor', 'k');
text(epoch(best_idx95)+0.5, best_mAP95+0.02, ...
    sprintf('Best: %.3f (Epoch %d)', best_mAP95, epoch(best_idx95)), ...
    'FontSize', 9, 'Color', [0.8 0.4 0]);

xlabel('Epoch'); ylabel('Score');
title('mAP 指标 (mAP Metrics)');
legend('mAP50', 'mAP50-95', 'Location', 'southeast');
grid on; set(gca, 'GridAlpha', 0.3);
ylim([0.3 1.0]);

sgtitle('SkyGuard AI - YOLOv11 无人机检测训练仪表板', 'FontSize', 15, 'FontWeight', 'bold');

%% 图4: 雷达图 - 最终模型 vs 第一轮
fig4 = figure('Name', '模型进步雷达图', 'Position', [300 300 500 500]);

categories = {'Precision', 'Recall', 'mAP50', 'mAP50-95'};
first_epoch = [precision(1), recall(1), mAP50(1), mAP50_95(1)];
last_epoch  = [precision(end), recall(end), mAP50(end), mAP50_95(end)];
best_epoch  = [precision(best_idx), recall(best_idx), mAP50(best_idx), mAP50_95(best_idx)];

% 归一化到0-1
first_norm = first_epoch;
last_norm  = last_epoch;
best_norm  = best_epoch;

angles = linspace(0, 2*pi, length(categories)+1)';

first_plot = [first_norm, first_norm(1)];
last_plot  = [last_norm, last_norm(1)];
best_plot  = [best_norm, best_norm(1)];

polarplot(angles, first_plot, '-o', 'LineWidth', 2, 'Color', colors(2,:)); hold on;
polarplot(angles, last_plot, '-s', 'LineWidth', 2, 'Color', colors(1,:));
polarplot(angles, best_plot, '-p', 'LineWidth', 2, 'Color', colors(5,:));
legend('Epoch 1', sprintf('Epoch %d (最新)', epoch(end)), ...
       sprintf('Epoch %d (最佳)', epoch(best_idx)), 'Location', 'southoutside');
title('模型性能进步对比', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'ThetaTick', linspace(0, 270, 4), 'ThetaTickLabel', categories);

%% 图5: 损失对比 (训练 vs 验证)
fig5 = figure('Name', '训练vs验证损失', 'Position', [150 150 900 350]);

subplot(1,3,1);
plot(epoch, train_box, '-', 'Color', colors(1,:), 'LineWidth', 2); hold on;
plot(epoch, val_box, '--', 'Color', colors(1,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('Box Loss');
legend('Train', 'Val', 'Location', 'northeast');
grid on; set(gca, 'GridAlpha', 0.3);

subplot(1,3,2);
plot(epoch, train_cls, '-', 'Color', colors(2,:), 'LineWidth', 2); hold on;
plot(epoch, val_cls, '--', 'Color', colors(2,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('Cls Loss');
legend('Train', 'Val', 'Location', 'northeast');
grid on; set(gca, 'GridAlpha', 0.3);

subplot(1,3,3);
plot(epoch, train_dfl, '-', 'Color', colors(3,:), 'LineWidth', 2); hold on;
plot(epoch, val_dfl, '--', 'Color', colors(3,:), 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title('DFL Loss');
legend('Train', 'Val', 'Location', 'northeast');
grid on; set(gca, 'GridAlpha', 0.3);

sgtitle('训练 vs 验证损失对比', 'FontSize', 14, 'FontWeight', 'bold');

%% 保存图表
saveas(fig1, 'runs/detect/models/trained/skyguard-v2-uav-3/chart_training_loss.png');
saveas(fig2, 'runs/detect/models/trained/skyguard-v2-uav-3/chart_validation_metrics.png');
saveas(fig3, 'runs/detect/models/trained/skyguard-v2-uav-3/chart_dashboard.png');
saveas(fig4, 'runs/detect/models/trained/skyguard-v2-uav-3/chart_radar.png');
saveas(fig5, 'runs/detect/models/trained/skyguard-v2-uav-3/chart_train_vs_val.png');

fprintf('\n========================================\n');
fprintf('  图表已保存至:\n');
fprintf('  runs/detect/models/trained/skyguard-v2-uav-3/\n');
fprintf('========================================\n');
fprintf('  1. chart_training_loss.png     - 训练损失曲线\n');
fprintf('  2. chart_validation_metrics.png - 验证指标曲线\n');
fprintf('  3. chart_dashboard.png          - 综合仪表板\n');
fprintf('  4. chart_radar.png              - 雷达图对比\n');
fprintf('  5. chart_train_vs_val.png       - 训练vs验证\n');
fprintf('========================================\n\n');

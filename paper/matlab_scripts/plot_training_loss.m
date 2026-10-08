% 训练损失收敛曲线绘制脚本
% 运行方式：在 Matlab 中 cd 到项目根目录，运行此脚本
% 输入数据：各模型 results.csv 文件
% 输出：训练损失收敛曲线图

clear; clc; close all;

% 定义模型路径和名称
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

markers = {'o', 's', '^', 'd', 'v'};

figure('Position', [100, 100, 1200, 800]);

for idx = 1:length(model_names)
    csv_data = readtable(model_paths{idx});
    epochs = csv_data.epoch;
    train_box = csv_data.train_box_loss;
    val_box = csv_data.val_box_loss;
    train_cls = csv_data.train_cls_loss;
    val_cls = csv_data.val_cls_loss;
    
    subplot(2, 1, 1);
    plot(epochs, train_box, '-', 'Color', colors(idx, :), 'LineWidth', 2, ...
         'Marker', markers{idx}, 'MarkerSize', 6, 'DisplayName', [model_names{idx} ' (train)']);
    hold on;
    plot(epochs, val_box, '--', 'Color', colors(idx, :), 'LineWidth', 2, ...
         'Marker', markers{idx}, 'MarkerSize', 6, 'DisplayName', [model_names{idx} ' (val)']);
    
    subplot(2, 1, 2);
    plot(epochs, train_cls, '-', 'Color', colors(idx, :), 'LineWidth', 2, ...
         'Marker', markers{idx}, 'MarkerSize', 6, 'DisplayName', [model_names{idx} ' (train)']);
    hold on;
    plot(epochs, val_cls, '--', 'Color', colors(idx, :), 'LineWidth', 2, ...
         'Marker', markers{idx}, 'MarkerSize', 6, 'DisplayName', [model_names{idx} ' (val)']);
end

subplot(2, 1, 1);
title('训练与验证 Box Loss 收敛曲线', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('训练轮数 (Epoch)', 'FontSize', 12);
ylabel('Box Loss', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);

subplot(2, 1, 2);
title('训练与验证 Classification Loss 收敛曲线', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('训练轮数 (Epoch)', 'FontSize', 12);
ylabel('Classification Loss', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);

sgtitle('各模型训练损失收敛曲线对比', 'FontSize', 16, 'FontWeight', 'bold');

print('paper/charts/training_loss_curves.png', '-dpng', '-r300');
disp('训练损失曲线图已保存到 paper/charts/training_loss_curves.png');

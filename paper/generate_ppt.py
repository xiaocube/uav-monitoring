#!/usr/bin/env python3
"""生成无人机反制项目汇报PPT"""
import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "paper"
OUTPUT_PATH = OUTPUT_DIR / "无人机智能监控系统_项目汇报.pptx"

BLUE_DARK = RGBColor(0x14, 0x28, 0x50)
GREEN_MID = RGBColor(0x28, 0x8C, 0x64)
GREEN_LIGHT = RGBColor(0x3C, 0xC8, 0x90)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
WHITE_SOFT = RGBColor(0xF0, 0xF5, 0xFA)
GRAY = RGBColor(0x80, 0x80, 0x80)


def add_title_slide(prs):
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "无人机智能监控系统"
    tf = title.text_frame
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(44)
            r.font.bold = True
            r.font.color.rgb = WHITE
            r.font.name = "黑体"
        p.alignment = PP_ALIGN.CENTER
    
    subtitle = slide.placeholders[1]
    subtitle.text = "基于机器视觉的低空安防解决方案\n\n项目研究汇报"
    tf = subtitle.text_frame
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(24)
            r.font.color.rgb = WHITE_SOFT
            r.font.name = "宋体"
        p.alignment = PP_ALIGN.CENTER
    
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BLUE_DARK


def add_agenda_slide(prs):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "汇报提纲"
    tf = title.text_frame
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(36)
            r.font.bold = True
            r.font.color.rgb = BLUE_DARK
            r.font.name = "黑体"
        p.alignment = PP_ALIGN.CENTER
    
    content = slide.placeholders[1]
    tf = content.text_frame
    tf.clear()
    
    items = [
        "1. 项目背景与研究意义",
        "2. 技术方案与系统架构",
        "3. 数据集构建与模型训练",
        "4. 实验结果与性能分析",
        "5. 应用场景与未来展望",
    ]
    
    for item in items:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(28)
        p.font.name = "宋体"
        p.font.color.rgb = BLUE_DARK
        p.space_after = Pt(12)


def add_background_slide(prs):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "项目背景与研究意义"
    tf = title.text_frame
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(36)
            r.font.bold = True
            r.font.color.rgb = BLUE_DARK
            r.font.name = "黑体"
        p.alignment = PP_ALIGN.CENTER
    
    content = slide.placeholders[1]
    tf = content.text_frame
    tf.clear()
    
    items = [
        "✈ 无人机技术快速发展，应用场景日益广泛",
        "⚠ \"黑飞\"现象严重威胁低空安全",
        "🎯 传统雷达监控成本高、误报率高",
        "💡 机器视觉提供低成本、高精度的解决方案",
    ]
    
    for item in items:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(24)
        p.font.name = "宋体"
        p.font.color.rgb = BLUE_DARK
        p.space_after = Pt(15)


def add_tech_slide(prs):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    p = title_box.text_frame.add_paragraph()
    p.text = "技术方案与系统架构"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "黑体"
    p.alignment = PP_ALIGN.CENTER
    
    tech_items = [
        "• YOLOv11 深度学习目标检测框架",
        "• Mosaic / Mixup 数据增强策略",
        "• 多尺度训练与自适应锚框机制",
        "• 切片推理提升小目标检测精度",
        "• ONNX 模型量化压缩部署",
        "• MPS 加速推理引擎",
    ]
    
    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(4.5), Inches(5))
    tf = left_box.text_frame
    tf.clear()
    
    for item in tech_items:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(22)
        p.font.name = "宋体"
        p.font.color.rgb = BLUE_DARK
        p.space_after = Pt(10)
    
    img_path = ROOT / "archive/v3-skyguard-uav-20260723/skyguard-v2-uav-3/val_results/val_batch0_pred.jpg"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(5), Inches(1.5), width=Inches(4.5))
    
    caption = slide.shapes.add_textbox(Inches(5), Inches(5.5), Inches(4.5), Inches(0.5))
    p = caption.text_frame.add_paragraph()
    p.text = "YOLOv11 实时检测效果"
    p.font.size = Pt(16)
    p.font.name = "宋体"
    p.font.color.rgb = GRAY


def add_dataset_slide(prs):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    p = title_box.text_frame.add_paragraph()
    p.text = "数据集构建"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "黑体"
    p.alignment = PP_ALIGN.CENTER
    
    img_path = ROOT / "data/UAV.yolov11/train/images/0001_jpg.rf.U75jAmVMl1zFe3OhS2Cy.jpg"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(0.5), Inches(1.5), width=Inches(4))
    
    data_info = [
        "📊 总样本量：8,318 张",
        "",
        "📁 训练集：5,821 张 (70%)",
        "📁 验证集：1,664 张 (20%)",
        "📁 测试集：833 张 (10%)",
        "",
        "🎯 目标类别：无人机 (1类)",
        "",
        "🌍 场景覆盖：户外空域、城市环境、复杂背景",
    ]
    
    right_box = slide.shapes.add_textbox(Inches(5), Inches(1.5), Inches(4.5), Inches(5))
    tf = right_box.text_frame
    tf.clear()
    
    for item in data_info:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(22)
        p.font.name = "宋体"
        p.font.color.rgb = BLUE_DARK


def add_training_slide(prs):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    p = title_box.text_frame.add_paragraph()
    p.text = "模型训练"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "黑体"
    p.alignment = PP_ALIGN.CENTER
    
    img_path = ROOT / "paper/charts/training_loss_curves.png"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(0.5), Inches(1.5), width=Inches(9))


def add_results_slide(prs):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    p = title_box.text_frame.add_paragraph()
    p.text = "实验结果与精度对比"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "黑体"
    p.alignment = PP_ALIGN.CENTER
    
    img_path = ROOT / "paper/charts/map_comparison.png"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(0.5), Inches(1.5), width=Inches(9))


def add_performance_slide(prs):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    p = title_box.text_frame.add_paragraph()
    p.text = "推理性能分析"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "黑体"
    p.alignment = PP_ALIGN.CENTER
    
    img_path = ROOT / "paper/charts/inference_performance.png"
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), Inches(0.5), Inches(1.5), width=Inches(9))


def add_metrics_slide(prs):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    p = title_box.text_frame.add_paragraph()
    p.text = "核心性能指标"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "黑体"
    p.alignment = PP_ALIGN.CENTER
    
    metrics = [
        ("mAP50", "96.6%"),
        ("Precision", "94.9%"),
        ("Recall", "94.9%"),
        ("FPS", "64.3"),
    ]
    
    spacing = Inches(9) / 4
    for i, (label, value) in enumerate(metrics):
        left = Inches(0.5) + i * spacing
        
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.8), spacing - Inches(0.3), Inches(2.5))
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE_SOFT
        box.line.color.rgb = GREEN_LIGHT
        box.line.width = Pt(2)
        
        tf = box.text_frame
        tf.clear()
        
        p = tf.add_paragraph()
        p.text = label
        p.font.size = Pt(20)
        p.font.name = "宋体"
        p.font.color.rgb = GREEN_MID
        p.alignment = PP_ALIGN.CENTER
        
        p = tf.add_paragraph()
        p.text = value
        p.font.size = Pt(40)
        p.font.bold = True
        p.font.name = "黑体"
        p.font.color.rgb = BLUE_DARK
        p.alignment = PP_ALIGN.CENTER


def add_scenarios_slide(prs):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    p = title_box.text_frame.add_paragraph()
    p.text = "应用场景"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "黑体"
    p.alignment = PP_ALIGN.CENTER
    
    scenes = [
        ("✈ 机场安防", "无人机入侵检测与预警"),
        ("🏛 重要场所", "政府机关、军事禁区监控"),
        ("🎡 大型活动", "演唱会、体育赛事低空管控"),
        ("🏭 工业厂区", "禁飞区无人机识别"),
    ]
    
    col_spacing = Inches(4.5)
    row_spacing = Inches(2.5)
    
    for i, (title_text, desc) in enumerate(scenes):
        row = i // 2
        col = i % 2
        left = Inches(0.5) + col * col_spacing
        top = Inches(1.8) + row * row_spacing
        
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, col_spacing - Inches(0.5), row_spacing - Inches(0.8))
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE_SOFT
        box.line.color.rgb = GREEN_LIGHT
        box.line.width = Pt(2)
        
        tf = box.text_frame
        tf.clear()
        
        p = tf.add_paragraph()
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.name = "黑体"
        p.font.color.rgb = GREEN_MID
        p.alignment = PP_ALIGN.CENTER
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(18)
        p.font.name = "宋体"
        p.font.color.rgb = BLUE_DARK
        p.alignment = PP_ALIGN.CENTER


def add_summary_slide(prs):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "总结与展望"
    tf = title.text_frame
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(36)
            r.font.bold = True
            r.font.color.rgb = BLUE_DARK
            r.font.name = "黑体"
        p.alignment = PP_ALIGN.CENTER
    
    content = slide.placeholders[1]
    tf = content.text_frame
    tf.clear()
    
    summary_items = [
        "✅ 完成了基于YOLOv11的无人机目标检测系统",
        "✅ 构建了8,000+样本的无人机检测数据集",
        "✅ 实现了96.6%的mAP50检测精度",
        "✅ 达到了64 FPS的实时推理速度",
        "",
        "🔮 轻量化嵌入式部署",
        "🔮 多目标持续跟踪",
        "🔮 复杂空域抗干扰识别",
        "🔮 新型无人机机型适配",
    ]
    
    for item in summary_items:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(24)
        p.font.name = "宋体"
        p.font.color.rgb = BLUE_DARK
        p.space_after = Pt(10)


def add_ending_slide(prs):
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "谢谢观看"
    tf = title.text_frame
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(48)
            r.font.bold = True
            r.font.color.rgb = WHITE
            r.font.name = "黑体"
        p.alignment = PP_ALIGN.CENTER
    
    subtitle = slide.placeholders[1]
    subtitle.text = "欢迎交流与指导"
    tf = subtitle.text_frame
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(28)
            r.font.color.rgb = WHITE_SOFT
            r.font.name = "宋体"
        p.alignment = PP_ALIGN.CENTER
    
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BLUE_DARK


def main():
    print("创建PPT文档...")
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    add_title_slide(prs)
    add_agenda_slide(prs)
    add_background_slide(prs)
    add_tech_slide(prs)
    add_dataset_slide(prs)
    add_training_slide(prs)
    add_results_slide(prs)
    add_performance_slide(prs)
    add_metrics_slide(prs)
    add_scenarios_slide(prs)
    add_summary_slide(prs)
    add_ending_slide(prs)
    
    prs.save(OUTPUT_PATH)
    print(f"✅ PPT已生成: {OUTPUT_PATH}")
    print(f"文件大小: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")
    print(f"页数: {len(prs.slides)}")


if __name__ == "__main__":
    main()

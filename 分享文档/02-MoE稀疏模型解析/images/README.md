# MoE架构图资源

本目录用于存放MoE相关论文的架构图。

## 需要的图片文件

1. **jacobs1991.png** - Jacobs 1991原始MoE架构图
   - 来源：Jacobs et al. "Adaptive Mixtures of Local Experts", Neural Computation, 1991
   - 提取方法：从论文PDF中截图或提取

2. **shazeer2017.png** - Shazeer 2017稀疏门控MoE架构图
   - 来源：Shazeer et al. "Outrageously Large Neural Networks", arXiv:1701.06538
   - 论文链接：https://arxiv.org/abs/1701.06538
   - 提取方法：从论文PDF的Figure 1和Figure 2中提取

3. **switch2021.png** - Switch Transformer架构图
   - 来源：Fedus et al. "Switch Transformers", arXiv:2101.03961
   - 论文链接：https://arxiv.org/abs/2101.03961
   - 提取方法：从论文PDF的Figure 1中提取

4. **mixtral2024.png** - Mixtral 8x7B架构图
   - 来源：Mixtral of Experts (Mistral AI, 2024)
   - 官方博客：https://mistral.ai/news/mixtral-of-experts/
   - 论文链接：https://arxiv.org/abs/2401.04088
   - 提取方法：从官方博客或论文中提取

5. **moe_transformer_layer.png** - MoE Transformer Layer完整架构图
   - 来源：综合多个资源
   - 参考：Hugging Face文档、Mixtral论文等

## 提取方法

### 方法1：从PDF中提取（推荐）

1. 下载论文PDF
2. 使用PDF阅读器（如Adobe Reader）打开
3. 找到架构图页面
4. 使用截图工具或PDF导出功能提取图片
5. 保存为PNG格式，放到本目录

### 方法2：从网页截图

1. 访问论文的arXiv页面或官方博客
2. 找到架构图
3. 使用浏览器截图工具保存
4. 保存为PNG格式，放到本目录

### 方法3：使用Python脚本提取（如果PDF已下载）

```python
import fitz  # PyMuPDF
import os

def extract_figures_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images()
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_dir}/page_{page_num}_img_{img_index}.{image_ext}"
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
    doc.close()

# 使用示例
# extract_figures_from_pdf("shazeer2017.pdf", "./images")
```

## 图片要求

- 格式：PNG（推荐）或JPG
- 分辨率：至少300 DPI，确保清晰度
- 命名：使用上述文件名
- 大小：建议宽度在800-1200像素之间



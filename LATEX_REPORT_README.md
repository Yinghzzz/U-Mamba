# LaTeX报告使用说明

## 📄 文件说明

已创建的LaTeX报告：
- **`brain_encoder_report.tex`** - 完整的作业报告（中英文混合，学术风格）

## 📋 报告内容结构

### 1. 摘要（Abstract）
- 研究背景和目标
- 方法概述
- 主要结果

### 2. 引言（Introduction）
- 跨模态生成背景
- U-Mamba简介
- 研究目标

### 3. 方法（Methods）
- 从分割到重建的任务改造
- 网络架构
- 损失函数设计
- 实现细节（数据准备、训练配置、多GPU训练）

### 4. 实验结果（Results）
- 重建质量评估（PSNR, SSIM, MAE）
- 训练过程分析
- 可视化结果
- 特征分析

### 5. 讨论（Discussion）
- 方法有效性分析
- 与分割任务对比
- 多GPU训练经验
- 局限性

### 6. 未来工作（Future Work）
- 下游任务验证
- 损失函数改进
- 多尺度特征融合
- 对比学习结合

### 7. 结论（Conclusion）
- 主要贡献总结
- 实验结果概括

### 8. 附录（Appendix）
- 代码实现
- 实验环境
- 使用命令

## 🔧 编译方法

### 方法1：使用XeLaTeX（推荐，支持中文）

```bash
# 编译LaTeX文档
xelatex brain_encoder_report.tex
xelatex brain_encoder_report.tex  # 第二次编译以生成目录和引用

# 如果需要参考文献
bibtex brain_encoder_report
xelatex brain_encoder_report.tex
xelatex brain_encoder_report.tex
```

### 方法2：使用PDFLaTeX

```bash
pdflatex brain_encoder_report.tex
bibtex brain_encoder_report.tex
pdflatex brain_encoder_report.tex
pdflatex brain_encoder_report.tex
```

### 方法3：使用Overleaf（在线编译）

1. 访问 https://www.overleaf.com/
2. 创建新项目
3. 上传 `brain_encoder_report.tex`
4. 设置编译器为 XeLaTeX
5. 点击 "Recompile"

### 方法4：使用VSCode + LaTeX Workshop

1. 安装VSCode扩展：LaTeX Workshop
2. 打开 `brain_encoder_report.tex`
3. 使用快捷键编译：
   - Windows/Linux: `Ctrl+Alt+B`
   - macOS: `Cmd+Option+B`

## 📦 所需LaTeX包

报告使用了以下LaTeX包（需要完整的TeX发行版）：

```latex
ctex          % 中文支持
amsmath       % 数学公式
amssymb       % 数学符号
graphicx      % 图片插入
geometry      % 页面设置
hyperref      % 超链接
booktabs      % 表格
multirow      % 表格多行
float         % 图表浮动
subfigure     % 子图
cite          % 引用
algorithm     % 算法
algorithmic   % 算法伪代码
listings      % 代码高亮
xcolor        % 颜色
```

### 安装TeX发行版

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
sudo apt-get install texlive-xetex
sudo apt-get install texlive-lang-chinese
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
- 下载并安装 MiKTeX: https://miktex.org/download
- 或者 TeX Live: https://www.tug.org/texlive/

## 🎨 自定义报告

### 1. 修改个人信息

在第44-47行修改：
```latex
\title{\textbf{基于U-Mamba的脑MRI特征提取研究} \\
\large 深度学习作业报告}
\author{姓名：XXX \quad 学号：XXXXXXXX}
\date{\today}
```

### 2. 插入实验结果

#### 插入训练曲线图
在第256-267行替换占位符：
```latex
% 取消注释并替换路径
\includegraphics[width=0.9\textwidth]{figures/progress.png}
```

**准备图片**：
```bash
# 创建figures目录
mkdir -p figures

# 复制训练曲线图
cp /path/to/nnUNet_results/.../progress.png figures/

# 或者从服务器下载
scp user@server:/path/to/progress.png figures/
```

#### 插入可视化结果
在第275-289行：
```latex
\subfigure[原始图像]{
    \includegraphics[width=0.3\textwidth]{figures/original.png}
}
\subfigure[重建图像]{
    \includegraphics[width=0.3\textwidth]{figures/reconstructed.png}
}
\subfigure[差异热图]{
    \includegraphics[width=0.3\textwidth]{figures/difference.png}
}
```

**准备可视化图片**：
```bash
# 从评估结果复制
cp evaluation_output/results/visualizations/sample_*.png figures/
```

### 3. 填写实验数据

#### 表格2：评估结果（第222行）
填写你的实际评估数据：
```latex
PSNR (dB) & 32.45 $\pm$ 2.31 & 32.78 & 优秀 \\
SSIM & 0.943 $\pm$ 0.012 & 0.945 & 优秀 \\
MAE & 0.042 $\pm$ 0.008 & 0.041 & 优秀 \\
```

从 `evaluation_results.json` 中获取数据：
```bash
cat evaluation_output/results/evaluation_results.json
```

#### 表格4：特征统计（第312行）
```latex
特征维度 & 512 \\
特征均值 & 0.123 \\
特征标准差 & 0.456 \\
```

### 4. 添加更多内容

#### 添加新的图片
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/your_figure.png}
    \caption{你的图片说明}
    \label{fig:your_label}
\end{figure}
```

#### 添加新的表格
```latex
\begin{table}[htbp]
    \centering
    \caption{表格标题}
    \label{tab:your_table}
    \begin{tabular}{lcc}
        \toprule
        \textbf{列1} & \textbf{列2} & \textbf{列3} \\
        \midrule
        数据1 & 数据2 & 数据3 \\
        \bottomrule
    \end{tabular}
\end{table}
```

#### 引用图表
```latex
如图\ref{fig:your_label}所示...
如表\ref{tab:your_table}所示...
```

## 📊 完整的工作流程

### Step 1: 收集实验数据

```bash
# 1. 训练曲线
cp ${RESULTS_DIR}/progress.png figures/

# 2. 运行评估
python evaluate_brain_encoder.py \
    --mode simple \
    --original_dir ./imagesTs \
    --predicted_dir ./predictions \
    --output_dir ./evaluation

# 3. 复制评估结果
cp evaluation/evaluation_results.json ./
cp evaluation/visualizations/sample_001.png figures/reconstruction.png
cp evaluation/metrics_distribution.png figures/
```

### Step 2: 提取数据填入LaTeX

```python
# 读取评估结果
import json
with open('evaluation_results.json') as f:
    results = json.load(f)

print("PSNR:", results['metrics']['psnr'])
print("SSIM:", results['metrics']['ssim'])
print("MAE:", results['metrics']['mae'])
```

### Step 3: 修改LaTeX文件

1. 打开 `brain_encoder_report.tex`
2. 修改个人信息（标题、姓名、学号）
3. 填入实验数据（表格中的数值）
4. 取消图片注释并设置正确路径
5. 保存文件

### Step 4: 编译生成PDF

```bash
# 编译两次以生成目录和交叉引用
xelatex brain_encoder_report.tex
xelatex brain_encoder_report.tex

# 生成的PDF文件
ls -lh brain_encoder_report.pdf
```

### Step 5: 检查结果

打开 `brain_encoder_report.pdf` 检查：
- ✅ 所有图片正确显示
- ✅ 表格数据完整
- ✅ 参考文献格式正确
- ✅ 页码和目录正确
- ✅ 无编译错误或警告

## 🐛 常见问题

### Q1: 中文显示乱码

**解决方案**：
- 使用 XeLaTeX 而不是 PDFLaTeX 编译
- 确保系统安装了中文字体
- 在Overleaf中设置编译器为XeLaTeX

### Q2: 找不到某个包

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra
sudo apt-get install texlive-fonts-extra

# 或者使用tlmgr
tlmgr install <package-name>
```

### Q3: 图片不显示

**解决方案**：
- 检查图片路径是否正确
- 确保 `figures/` 目录存在
- 检查图片格式（支持PDF, PNG, JPG）
- 取消注释图片插入命令

### Q4: 编译很慢

**解决方案**：
- 第一次编译会慢（需要生成辅助文件）
- 后续编译会快很多
- 使用 `pdflatex -interaction=nonstopmode` 跳过错误

### Q5: 参考文献不显示

**解决方案**：
需要完整的编译流程：
```bash
xelatex brain_encoder_report.tex
bibtex brain_encoder_report
xelatex brain_encoder_report.tex
xelatex brain_encoder_report.tex
```

## 📚 快速参考

### LaTeX数学公式
```latex
行内公式: $E = mc^2$
行间公式:
\begin{equation}
    E = mc^2
\end{equation}
```

### LaTeX表格
```latex
\begin{table}[htbp]
    \centering
    \begin{tabular}{|l|c|r|}
        \hline
        左对齐 & 居中 & 右对齐 \\
        \hline
        数据1 & 数据2 & 数据3 \\
        \hline
    \end{tabular}
\end{table}
```

### LaTeX图片
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.5\textwidth]{image.png}
    \caption{图片说明}
\end{figure}
```

### 代码块
```latex
\begin{lstlisting}[language=Python]
def hello():
    print("Hello, World!")
\end{lstlisting}
```

## 💡 提示

1. **先填占位符，再编译**：确保所有 `XX.XX` 和 `XXX` 都被替换
2. **保留备份**：修改前先备份原始文件
3. **逐步编译**：每次修改一小部分后编译，便于定位错误
4. **使用注释**：暂时不用的内容用 `%` 注释掉
5. **图片格式**：优先使用PDF或PNG格式，避免JPG（可能有压缩伪影）

## 📖 进阶自定义

### 修改页面布局
```latex
\geometry{
    left=2.5cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm
}
```

### 修改字体大小
```latex
\documentclass[12pt,a4paper]{article}  % 12pt更大
```

### 添加页眉页脚
```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhead[L]{左页眉}
\fancyhead[R]{右页眉}
```

### 修改代码高亮主题
```latex
\lstset{
    backgroundcolor=\color{lightgray},
    keywordstyle=\color{blue}\bfseries,
    commentstyle=\color{green!60!black},
}
```

## ✅ 检查清单

提交前确认：
- [ ] 个人信息已修改（姓名、学号）
- [ ] 所有实验数据已填写
- [ ] 所有图片已插入并显示正常
- [ ] 表格数据完整准确
- [ ] 无编译错误或警告
- [ ] 参考文献格式正确
- [ ] PDF生成成功
- [ ] 全文通读无错别字
- [ ] 页码连续无缺失
- [ ] 图表编号正确

## 📞 获取帮助

如果遇到问题：
1. 查看LaTeX编译日志（.log文件）
2. 搜索具体错误信息
3. 访问 https://tex.stackexchange.com/
4. 使用Overleaf的帮助文档

---

**祝编译顺利！如有问题欢迎反馈。**

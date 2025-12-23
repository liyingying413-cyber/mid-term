# Interactive Poster Generator (Streamlit)

一个可交互的艺术海报生成器：
- 滑块控制：层数、形状、扩散度、wobble、透明度、阴影、描边等
- 多种内置色彩模式
- 支持上传 CSV 调色板（name,r,g,b；r/g/b 为 0~1 浮点）
- 导出 300 DPI PNG

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py

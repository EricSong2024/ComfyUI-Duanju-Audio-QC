# ComfyUI-Duanju-Audio-QC

短剧 Skill 自有的 ComfyUI 音频质检节点。`DuanjuCLAPBGMSelector` 可接收最多六个候选音频，结合 CLAP 正负文本相似度和 DSP 指标选出 BGM；候选全部不达标时输出安全静音，并返回 JSON 报告。

## 安装

将仓库克隆到 `ComfyUI/custom_nodes/ComfyUI-Duanju-Audio-QC`，然后在 ComfyUI 的 Python 环境中安装：

```bash
python -m pip install -r requirements.txt
```

CLAP 模型必须由用户事先放在 `ComfyUI/models/clap/clap-htsat-unfused/`。节点不会自动下载模型。

安装或更新后重启 ComfyUI，再检查 `/object_info/DuanjuCLAPBGMSelector`。

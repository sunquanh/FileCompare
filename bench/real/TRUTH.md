# 真实样本真值口径

`bench/real/` 里是真图和人工/人工修订过的真值掩膜，体积大且属于用户素材，
所以不入库（见根目录 `.gitignore`）。入库的只有两样文本：

- `manifest.json`：每个用例的尺寸、掩膜像素数、掩膜包围盒、两份文件的
  sha256，由 `python tools/wm_truth.py --update` 生成。
- 本文件：真值从哪来、画到哪算。

真值只在本机时，基准里的 `hit` / `damage` 一旦变化，无法区分是算法改了还是
真值改了。第 10 批之前就踩过：`real_screen_ide` 换成人工真值后 `hit` 从
0.88 掉到 0.33，报表上只是数字变了。现在跑评测会先核对清单，素材被动过就
先打一行警告，提示 `hit` 与旧基线不可比。

## 规矩

1. **真值不来自检测器。** 掩膜由人工描笔画、或对检测初稿逐块补漏/删多余
   得到，检测器的输出只能当草稿，草稿要留档（`mask_detect_draft.png`）而不
   覆盖真值。
2. **改真值要写清口径。** 换真值必须改 `meta.json` 的 `truth` 字段，并在
   下面表格里加一行说明，然后 `--update` 重写清单、重跑基线。
3. **水印笔画外的抗锯齿边不由真值裁定。** 笔画外圈的柔边算"掩膜多检"，
   进 `damage` 的观察范围；真值只标笔画能看见的部分，避免真值自己变严格。

## 用例

| 用例 | 类别 | 真值来源 | 说明 |
| --- | --- | --- | --- |
| `real_screen_ide` | screen（平铺斜带） | `manual-band` | 手量斜带 ∪ 偏黄量，不含 `detect_tint` 或掩膜的任何输出；旧检测初稿另存 `mask_detect_draft.png` |
| `real_doc_xlsx` | doc | `manual-mask`（缺省） | 检测初稿 + 人工补漏/删多余 |
| `clean_photo_01..03` | photo（无水印） | `empty` | 空掩膜，只用于探测假阳性（`fp` 必须为 0） |

## 用法

```bash
python tools/wm_truth.py --verify   # 核对本地素材与清单（不一致退出码 1）
python tools/wm_truth.py --update   # 真值确有意改动后重写清单

PYTHONIOENCODING=utf-8 python tools/wm_bench.py bench/real \
    --baseline bench/real_baseline.json
```

`bench/real_baseline.json` 同样是本地文件：真值变了它也要跟着重存，否则比较
的就是两套尺子。

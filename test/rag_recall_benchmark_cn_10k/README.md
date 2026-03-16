# 中文 RAG 召回评测集（10,000 Queries）

这是一个用于中文 RAG / 检索召回评估的合成高质量测试集，适合评估：
- 向量召回 / BM25 / 混合检索
- reranker 前后的 Recall@k、MRR、nDCG
- chunk 粒度、召回路数、hard negative 抗干扰能力

## 文件说明
- `corpus.jsonl`：2500 条知识库语料，每条为一个可检索 chunk
- `queries.jsonl`：10000 条查询，每条含标准正例文档、query_type、difficulty、expected_answer、hard negatives
- `qrels.tsv`：标准相关性标注
- `benchmark_pairs_10000.csv`：便于直接查看和导入 Excel 的平铺版
- `hard_negatives.jsonl`：每条查询的 5 个困难负例
- `evaluate_recall.py`：简单评测脚本

## 数据特征
- 语料数：2500
- 查询数：10000
- 类别数：24
- 子类型数：25

### Query Type 分布
- numeric_limit: 1800
- exact_fact: 2400
- temporal: 2000
- scenario: 2500
- threshold: 700
- procedural: 300
- alias: 100
- exception: 200

### Difficulty 分布
- easy: 4200
- medium: 3100
- hard: 2700

## 建议评测方式
1. 先用 `queries.jsonl` 中的 `query` 去检索 `corpus.jsonl`。
2. 用返回的 Top-K 与 `qrels.tsv` 比较，计算 Recall@1/5/10/20。
3. 若有重排序模型，可先召回 50~200 条，再 rerank，比较前后增益。
4. 若想测 hard negative 鲁棒性，可观察正例是否被同 subtype 的近似干扰文档压制。

## 注意
- 这是面向 RAG 检索链路的测试集，不是开放域百科问答集。
- 语料风格偏企业知识库 / 制度 / 产品说明，适合中文业务型 RAG。
- 如需更贴近你的项目，我可以在下一步按你的行业改成：法律 / 医疗 / 电商 / 制造 / 矿山 / 学术论文 / 专利文档风格。
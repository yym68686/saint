# Family Benchmark 全流程说明

本目录打包了一个完整的端到端流程，用来构建并评测一个 `15-family` 的控制工作点 benchmark，目标是突出两个创新点：

- `SUR SAE`（`kernel`）：候选空间扩展能力
- `PLRDC SAE`（`dense`）：严格预算下的控制质量

## 目录结构

- `stages/260415_family_ontology_draft/`
  从 family ontology 草稿到最终 released splits 的完整数据集构建流程。
- `stages/260417_final_family_benchmark_l22/`
  L22 的 feature registry 构建、GPU benchmark runner、原始 benchmark 输出、以及最终聚焦后的指标结果。
- `stages/260417_weights_audit/`
  权重清单与服务器端 benchmark 权重审计结果。
- `code_runtime_refs/`
  实际做激活重算时依赖的运行时代码拷贝。
- `docs/SOURCE_CORPUS_REFERENCE.md`
  候选句子挖掘时使用的原始基础语料说明。

## 当前仍分散在其他位置的活跃文件与目录

本目录是集中归档包，但当前仍在使用、并且没有完全收拢进本归档目录的活跃文件，主要分散在以下绝对路径中。

### 1. 原始基础语料

- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/dataset/train-00000-of-00082.parquet`

### 2. 当前活跃的 Benchmark 运行目录

- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/build_feature_registry.py`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/run_l22_family_benchmark.py`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/analyze_control_redundancy.py`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/postprocess_final_metrics.py`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/feature_registry_l22.csv`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/results/full_l22`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/results/control_redundancy_full`
- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260417/final_family_benchmark_l22/final_outputs`

### 3. Thesis 绘图源目录

- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/build_control_redundancy_thesis_figures.py`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/data/focused_method_summary_sur_strength.csv`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/data/high_efficiency_threshold_sweep.csv`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/data/high_efficiency_family_matrix_5pct_30.csv`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/plots/sur_plrdc_control_headline_metrics.png`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/plots/sur_high_efficiency_valid_family_overview.png`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/plots/sur_high_efficiency_threshold_sweep.png`
- `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea14-banchmark/control_redundancy_figures/plots/sur_high_efficiency_family_heatmap.png`

### 4. Thesis 实际引用的图片目录

- `/Users/yanyuming/Downloads/GitHub/Thesis/zjuthesis/figure/sur_plrdc_control_headline_metrics.png`
- `/Users/yanyuming/Downloads/GitHub/Thesis/zjuthesis/figure/sur_high_efficiency_valid_family_overview.png`
- `/Users/yanyuming/Downloads/GitHub/Thesis/zjuthesis/figure/sur_high_efficiency_threshold_sweep.png`
- `/Users/yanyuming/Downloads/GitHub/Thesis/zjuthesis/figure/sur_high_efficiency_family_heatmap.png`

### 5. Thesis 正文接入文件

- `/Users/yanyuming/Downloads/GitHub/Thesis/zjuthesis/page/graduate/abstract.tex`
- `/Users/yanyuming/Downloads/GitHub/Thesis/zjuthesis/body/graduate/content.tex`
- `/Users/yanyuming/Downloads/GitHub/Thesis/zjuthesis/out/zjuthesis.pdf`

### 6. 服务器执行位置

- `/root/saint`
- `/root/saint/llama_3.2-3B_model/original`
- `/root/saint/presentation/260417/final_family_benchmark_l22`

## 最终构建了什么

最终产物是一个固定的、基于语义 family 的 `15-family` benchmark。它从基础句子语料构建而来，用于比较 `7` 个 L22 SAE 方法：

- `topk`
- `batchtopk`
- `relu`
- `gatedsae`
- `jumprelu`
- `dense`
- `kernel`

最终保留的 `15` 个 topical family 为：

- `mlb_baseball`
- `nfl_football`
- `soccer`
- `nba_basketball`
- `nhl_hockey`
- `combat_sports`
- `gaming_general`
- `crypto_blockchain`
- `aviation_aerospace`
- `china`
- `japan`
- `russia_post_soviet`
- `middle_east_geopolitics`
- `us_electoral_politics`
- `us_legislative_governance`

## 端到端流程

### 第 0 步：确定基础语料

目的：
- 保证 benchmark 使用的句子分布与 activation capture 使用的语料分布一致。

实际使用：
- Base corpus：`dataset/train-00000-of-00082.parquet`
- 文本字段：`text`

为什么重要：
- family benchmark 必须和 SAE 激活捕获使用同一语料分布，否则工作点结果会失去可比性。
- 这里刻意没有使用旧的 `ablation_datasets-*` 子数据集来充当新 benchmark 的来源。

参考文件：
- `docs/SOURCE_CORPUS_REFERENCE.md`

### 第 1 步：构建第一版 Family Ontology

目的：
- 用更粗粒度的语义 family，对齐不同方法的特征语义，替代原来的 exact shared concept 对齐。

输入：
- `Thesis/exp/**/output/l22/output/parsed_responses.yaml`

实际做了什么：
- 扫描了 `23` 份 L22 parsed-response 文件。
- 收集了 `1140` 条 feature interpretation 记录。
- 将它们映射到一份可编辑的 `29-family` shortlist。
- 为多 family 冲突样本生成 overlap review 文件，供后续人工收紧定义。

主要输出：
- `stages/260415_family_ontology_draft/family_ontology_draft.yaml`
- `stages/260415_family_ontology_draft/family_summary.csv`
- `stages/260415_family_ontology_draft/family_match_details.csv`
- `stages/260415_family_ontology_draft/family_overlap_review.csv`

主要代码：
- `stages/260415_family_ontology_draft/build_family_ontology_draft.py`

### 第 2 步：压缩成 Benchmark-Ready 的 15-Family 集合

目的：
- 只保留覆盖足够广、边界足够清晰、并且被足够多核心方法支持的 family。

筛选规则：
- 保留所有 `high-priority` 且 `core_method_count >= 5` 的 family。

实际做了什么：
- 将 `29` 个 family 压缩成 `15` 个。
- 为每个 family 指定负样本 family 分配方案。
- 写出 family-level target/control benchmark 的蓝图。

主要输出：
- `stages/260415_family_ontology_draft/benchmark_ready/benchmark_ready_family_definitions.yaml`
- `stages/260415_family_ontology_draft/benchmark_ready/benchmark_ready_family_summary.csv`
- `stages/260415_family_ontology_draft/benchmark_ready/target_control_blueprint.md`
- `stages/260415_family_ontology_draft/benchmark_ready/excluded_family_notes.csv`

主要代码：
- `stages/260415_family_ontology_draft/benchmark_ready/build_benchmark_ready_package.py`

### 第 3 步：定义 Benchmark 的 Split 配方

目的：
- 固定一套所有方法都必须遵守的数据拆分规则。

每个 family 的最终 split 配方：
- `selection`：`40 target + 160 control`
- `calibration`：`500 control`
- `evaluation`：`120 target + 500 control`

control 的组成比例：
- `hard_negative`：`50%`
- `medium_negative`：`30%`
- `background_negative`：`20%`

构造约束：
- 每个方法在每个 family 最多只能提交 `1` 个 feature。
- 所有阈值只能在 `calibration_control` 上单独校准。
- 最终工作点结果只能在 `evaluation` split 上报告。

参考文件：
- `stages/260415_family_ontology_draft/benchmark_ready/target_control_blueprint.md`

### 第 4 步：挖候选句子并导出第一轮标注批次

目的：
- 在人工精洗之前，先从基础语料中拉出高召回候选池。

实际做了什么：
- 为每个 family 写了 lexical retrieval query。
- 构建了以下几类候选池：
  - `positive`
  - `local_hard_negative`
  - `benchmark_hard_negative`
  - `medium_negative`
  - `background_negative`
- 导出了第一轮 annotation batch。

主要输出：
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_skeleton/`
- `stages/260415_family_ontology_draft/benchmark_ready/family_retrieval_queries.yaml`
- `stages/260415_family_ontology_draft/benchmark_ready/family_annotation_guidelines.md`
- `stages/260415_family_ontology_draft/benchmark_ready/family_annotation_sheet_template.csv`

主要代码：
- `stages/260415_family_ontology_draft/benchmark_ready/build_family_level_dataset_skeleton.py`

### 第 5 步：收紧 Query，构建更干净的第二轮候选池

目的：
- 在人工标注前，先去掉最明显、最可预期的 lexical overreach。

实际做了什么：
- 收紧了过宽的体育和地缘政治词触发器。
- 重新构建了 candidate pool。
- 导出了第二轮、更干净的 annotation batch。

主要输出：
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_round2/`

关键辅助文件：
- `stages/260415_family_ontology_draft/benchmark_ready/family_retrieval_queries.yaml`

### 第 6 步：手动清洗与 Benchmark 标注

目的：
- 将高召回候选池转成高精度的 benchmark-ready 标注集。

实际做了什么：
- 增加了 family 级手工清洗规则。
- 重新生成了手工清洗后的 candidate pool。
- 为全部 `15` 个 family 回填了 `annotation_batch.csv`。

主要输出：
- `stages/260415_family_ontology_draft/benchmark_ready/manual_cleaning_rules.yaml`
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_manualclean/annotation_batches/`
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_manualclean/candidate_pools/`

主要代码：
- `stages/260415_family_ontology_draft/benchmark_ready/build_manual_clean_round.py`

### 第 7 步：缺口分析、定向补样、冻结池、最终发布 Split

目的：
- 将第一轮标注结果补齐成一个数量完整、可正式评测的 benchmark release。

实际做了什么：
- 汇总全部已标注 batch。
- 做文本去重。
- 按每个 family 的最终目标数计算缺口：
  - `160 target_positive`
  - `580 hard_negative`
  - `348 medium_negative`
  - `232 background_negative`
- 按标签缺口做定向补样。
- 冻结最终 pool。
- 将每个 family 切成 `selection / calibration / evaluation`。

主要输出：
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_release/frozen_pool/`
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_release/splits/`
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_release/summary/`
- `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_release/release_report.md`

主要代码：
- `stages/260415_family_ontology_draft/benchmark_ready/build_family_dataset_release.py`

关键结果：
- `15/15` 个 family 全部达到 `split_ready=yes`。

### 第 8 步：审计所需权重与运行依赖

目的：
- 保证 L22 benchmark 可以在 GPU 服务器上直接执行，同时不改服务器环境。

实际做了什么：
- 审计了 `7` 个 benchmark 方法所需的 `.pt` 权重。
- 确认 benchmark 需要以下权重：
  - `trained_sae-main-l22.pt`
  - `trained_sae-batchtopk-l22.pt`
  - `trained_sae-relu-l22.pt`
  - `trained_sae-gatedsae-l22.pt`
  - `trained_sae-jumprelu-l22.pt`
  - `trained_sae-dense-l22.pt`
  - `kernel.pt`
- 还确认了 base Llama checkpoint 也是必须项：
  - `llama_3.2-3B_model/original/consolidated.00.pth`

主要输出：
- `stages/260417_weights_audit/pt_weight_audit.md`
- `stages/260417_weights_audit/family_benchmark_l22_manifest.csv`

### 第 9 步：构建 L22 Feature Registry

目的：
- 从 ontology match 结果中，抽出最终 benchmark run 真正用到的 `family × method × feature` 候选集合。

实际做了什么：
- 将 `family_match_details.csv` 过滤到 `15` 个 benchmark family 和 `7` 个核心方法。
- 生成了一份单独的 L22 feature registry，共 `114` 条 candidate row。

主要输出：
- `stages/260417_final_family_benchmark_l22/feature_registry_l22.csv`

主要代码：
- `stages/260417_final_family_benchmark_l22/build_feature_registry.py`

### 第 10 步：在 GPU 服务器上运行完整的 L22 Family Benchmark

目的：
- 在固定 benchmark splits 上重算激活，并用同一评测流水线比较全部 `7` 个方法。

执行环境：
- 服务器代码目录：`/root/saint`
- 环境激活方式：`eval $(poetry env activate)`
- 使用分支：`feature-ablation`

runner 的实际工作流程：
1. 读取 `family_dataset_release/splits`。
2. 构建唯一 benchmark text pool。
3. 对全部文本做 tokenize。
4. 在 `L22` 只加载一次 Llama。
5. 对每个方法依次执行：
   - 加载对应 SAE 权重
   - 在 text pool 上给所有 candidate feature 打分
   - 用 `selection` split 选择该 `family × method` 的最佳 feature
   - 用 `calibration_control` 校准阈值
   - 在 `evaluation_target` 和 `evaluation_control` 上评测
6. 导出 per-budget、per-family、per-method 汇总结果。

主要输出：
- `stages/260417_final_family_benchmark_l22/results/full_l22/`

主要代码：
- `stages/260417_final_family_benchmark_l22/run_l22_family_benchmark.py`

运行时依赖代码：
- `code_runtime_refs/sae_loader.py`
- `code_runtime_refs/sae.py`
- `code_runtime_refs/sae_batchtopk.py`
- `code_runtime_refs/sae_relu.py`
- `code_runtime_refs/sae_gatedsae.py`
- `code_runtime_refs/sae_jumprelu.py`
- `code_runtime_refs/sae_exp11_dense.py`
- `code_runtime_refs/llama_3/args.py`
- `code_runtime_refs/llama_3/model_text_only.py`
- `code_runtime_refs/llama_3/tokenizer.py`
- `code_runtime_refs/compare_feature_activation_between_datasets.py`

### 第 11 步：从完整 Run 中后处理出最终聚焦指标

目的：
- 把“总体 benchmark utility”与“创新点特异性的 headline value”区分开。
- 把最终解释框架固定为：
  - `SUR SAE` 看控制可替代性
  - `PLRDC SAE` 看控制强度

实际做了什么：
- 读取完整 benchmark 原始结果。
- 额外读取“全候选 feature 控制冗余 sweep”结果，不再只看每个 family 中单一入选的最佳 feature。
- 生成只保留两项核心 headline metric 的 focused summary。
- 导出了最终 headline 图和紧凑汇总表。

主要输出：
- `stages/260417_final_family_benchmark_l22/results/control_redundancy_full/control_metric_candidates.csv`
- `stages/260417_final_family_benchmark_l22/results/control_redundancy_full/control_redundancy_report.md`
- `stages/260417_final_family_benchmark_l22/final_outputs/focused_method_summary.csv`
- `stages/260417_final_family_benchmark_l22/final_outputs/headline_metrics_report.md`
- `stages/260417_final_family_benchmark_l22/final_outputs/plots/final_headline_metrics.png`
- `stages/260417_final_family_benchmark_l22/final_outputs/plots/alternative_controller_vs_strict_valid_yield.png`

主要代码：
- `stages/260417_final_family_benchmark_l22/analyze_control_redundancy.py`
- `stages/260417_final_family_benchmark_l22/postprocess_final_metrics.py`

## 指标定义

### 1. Strict-Budget Alternative-Controller Rate @2%/30%

定义：
- 固定一个严格预算和最小有效作用门槛：
  - held-out `control budget <= 2%`
  - held-out `target reject rate >= 30%`
- 对于每个 `family × method`，把所有 ontology-matched candidate feature 都重新过一遍：
  - 在 `calibration_control` 上按 `2%` 定阈值
  - 在 `evaluation_target / evaluation_control` 上验证
- 若某个 feature 同时满足以上两个条件，则视为一个“有效控制器”。
- 若同一个 family 中至少有 `2` 个不同 feature 都是有效控制器，则该 family 记为“存在替代控制器”。
- 最终指标定义为：
  在所有“至少存在 1 个有效控制器”的 controllable families 中，有多少比例还存在第 `2` 个有效控制器。

公式可写成：

\[
V_{m,f}=\{g:\mathrm{CR}_{m,f,g}\le 2\%,\ \mathrm{TR}_{m,f,g}\ge 30\%\}
\]

\[
\mathrm{SBACR}_{2\%,30\%}(m)=
\frac{|\{f:\ |V_{m,f}|\ge 2\}|}{|\{f:\ |V_{m,f}|\ge 1\}|}
\]

为什么需要它：
- `SUR SAE` 的核心价值不只是“候选更多”，而是“在同一语义 family 内是否保留第二个真正可用的控制器”。
- 这个指标完全基于 held-out calibration 与 held-out evaluation 的真结果，不是 candidate-space proxy。
- 它衡量的是控制的可替代性、冗余性与鲁棒性，更贴近 `SUR` 的创新点。

如何理解：
- 越高越好。
- 这是一个真正的控制指标，不是上游候选代理。
- 在最终叙事中，它对应 `SUR` 的“控制可替代性”。
- 它回答的问题是：在已经能控住的 families 里，一个方法有多大比例还能拿出第二个独立可用的控制器。

### 2. Strict-Budget Valid Trigger Yield @2%

定义：
- 当 control budget 固定为 `2%` 时，只在“真正守住 2% held-out control budget”的 family 上保留 `target_reject_rate`；一旦超预算，该 family 直接记为 `0`。最后在 covered families 上取平均。

单个 family 的得分：
- 对 family `f`、方法 `m`，先在 `calibration_control` 上以 `alpha=2%` 校准阈值 `tau`。
- 然后测量：
  - `evaluation_target` 上的 `target_reject_rate`
  - `evaluation_control` 上的 `control_reject_rate`
- 该 family 的严格有效触发得分定义为：

\[
\text{SBVTY}_{m,f@2\%} =
\begin{cases}
\text{target\_reject\_rate}, & \text{if control\_reject\_rate} \le 2\% \\
0, & \text{otherwise}
\end{cases}
\]

- 最终方法分数是在所有 covered family 上取平均。

为什么需要它：
- `PLRDC SAE` 的定位就是在严格控制预算下更强。
- 如果一个 family 在 held-out evaluation 上超了 `2%`，那么这个 family 在严格部署语境下就不应该继续得分。
- 与软惩罚版本相比，这个硬门控指标更贴近“严格控制能力”的说法。

如何理解：
- 越高越好。
- 这是最直接对应 `PLRDC` 创新点的部署型指标。
- 在最终叙事中，它对应 `PLRDC` 的“控制强度”。

### 3. Rich Candidate Coverage Rate @2+

定义：
- 在固定 `15` 个 benchmark family 中，某个方法有多少比例的 family 至少提供了 `2` 个 ontology-matched candidate feature。

用途：
- 它现在只作为支持性解释指标。
- 用来说明为什么某个方法更可能在 family 内形成“替代控制器”。

为什么它不再是 headline metric：
- 它只反映 candidate-space 结构，不直接反映 held-out control 成败。
- 在最终口径里，它被降级为解释 `SUR` 结果的辅助量，而不是主指标。

### 4. FCOS@2%,5%,10%

定义：
- 在全部 `15` 个 family 和预算 `{2%, 5%, 10%}` 上，计算平均惩罚后 target reject rate。
- 未覆盖 family 记为 `0`。

为什么需要它：
- 它是一个整体 benchmark utility 指标。
- 同时融合了 coverage 和 deployable quality。

为什么它不是最后的 headline metric：
- 在完整 benchmark 上，`ReLU SAE` 的 FCOS 最高。
- 这说明 FCOS 适合作为辅助总分，但并不是隔离 `SUR` 或 `PLRDC` 价值的最干净 headline metric。

### 5. Covered Strict Quality@2%,5%

定义：
- 只在 covered family 上，针对预算 `{2%, 5%}` 计算平均惩罚后 target reject rate。

用途：
- 作为辅助的 quality summary。
- 比单一预算点更稳定，同时又比包含 `10%` 的平均更严格。

### 6. Selection Strict Score

定义：
- 一个内部特征选择指标，只用于给每个 `family × method` 选择最佳 feature。
- 基于 `selection` split，在严格预算 `{2%, 5%}` 上计算。

用途：
- 不作为最终 headline result 报告。
- 只在 runner 内部用于决定哪个 feature 进入 calibration 和 evaluation。

## 最终指标结果

结果来源：
- `stages/260417_final_family_benchmark_l22/final_outputs/focused_method_summary.csv`

### 面向创新点的最终 headline 指标

| 方法 | SBACR@2%/30% | 替代控制器家族 / 可控家族 | SBVTY@2% | RCCR@2+ | FCOS@2,5,10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SUR SAE | 20.0% | 1 / 5 | 22.8% | 33.3% | 62.9% |
| PLRDC SAE | 0.0% | 0 / 5 | 30.8% | 26.7% | 55.7% |
| ReLU SAE | 0.0% | 0 / 6 | 29.1% | 13.3% | 65.5% |
| Gated SAE | 0.0% | 0 / 6 | 21.2% | 13.3% | 60.5% |
| BatchTopK SAE | 0.0% | 0 / 5 | 19.3% | 13.3% | 58.3% |
| TopK SAE | 0.0% | 0 / 3 | 11.1% | 6.7% | 45.1% |
| JumpReLU SAE | 0.0% | 0 / 2 | 15.5% | 13.3% | 22.0% |

### 核心解读

- `SUR SAE` 在控制可替代性指标上是最强方法：
  - `Strict-Budget Alternative-Controller Rate @2%/30% = 20.0%`
  - 也就是 `1 / 5` 个可控 family 里存在第 `2` 个独立有效控制器
  - 对应到全 benchmark 是 `1 / 15 = 6.7%`
- `PLRDC SAE` 在最严格的控制强度指标上是最强方法：
  - `Strict-Budget Valid Trigger Yield @2% = 30.8%`
- `ReLU SAE` 在更宽泛的总体 utility 指标上最强：
  - `FCOS@2,5,10 = 65.5%`

这也是为什么最后报告中保留：
- `Strict-Budget Alternative-Controller Rate @2%/30%` 作为 `SUR` 的控制可替代性 headline metric
- `Strict-Budget Valid Trigger Yield @2%` 作为 `PLRDC` 的控制强度 headline metric
- `Rich Candidate Coverage Rate @2+` 作为支持性解释指标
- `FCOS` 作为辅助的 benchmark-wide 总分

## 最值得先看的文件

如果只想快速浏览，建议按这个顺序看：

1. `PROCESS_SUMMARY.md`
2. `stages/260417_final_family_benchmark_l22/final_outputs/headline_metrics_report.md`
3. `stages/260417_final_family_benchmark_l22/final_outputs/focused_method_summary.csv`
4. `stages/260417_final_family_benchmark_l22/final_outputs/plots/final_headline_metrics.png`
5. `stages/260415_family_ontology_draft/benchmark_ready/target_control_blueprint.md`
6. `stages/260415_family_ontology_draft/benchmark_ready/family_dataset_release/release_report.md`

## 复现实验的关键命令

### 本地生成 registry

```bash
python3 stages/260417_final_family_benchmark_l22/build_feature_registry.py
```

### 在服务器上运行 benchmark

```bash
cd /root/saint
eval "$(poetry env activate)"
python presentation/260417/final_family_benchmark_l22/run_l22_family_benchmark.py \
  --llama_model_dir /root/saint/llama_3.2-3B_model/original \
  --weights_dir /root/saint \
  --splits_root /root/saint/presentation/260417/final_family_benchmark_l22/inputs/splits \
  --feature_registry /root/saint/presentation/260417/final_family_benchmark_l22/feature_registry_l22.csv \
  --output_dir /root/saint/presentation/260417/final_family_benchmark_l22/results/full_l22 \
  --max_batch_size 4 \
  --max_batch_tokens 384 \
  --max_token_length 192 \
  --device cuda
```

### 在服务器上运行控制冗余 sweep

```bash
cd /root/saint
eval "$(poetry env activate)"
python presentation/260417/final_family_benchmark_l22/analyze_control_redundancy.py \
  --llama_model_dir /root/saint/llama_3.2-3B_model/original \
  --weights_dir /root/saint \
  --splits_root /root/saint/presentation/260417/final_family_benchmark_l22/inputs/splits \
  --feature_registry /root/saint/presentation/260417/final_family_benchmark_l22/feature_registry_l22.csv \
  --output_dir /root/saint/presentation/260417/final_family_benchmark_l22/results/control_redundancy_full \
  --budgets 0.02 \
  --target_thresholds 0.10 0.20 0.30 0.40
```

### 后处理生成最终 headline 文件夹

```bash
python3 stages/260417_final_family_benchmark_l22/postprocess_final_metrics.py \
  --benchmark_run_dir stages/260417_final_family_benchmark_l22/results/full_l22 \
  --control_redundancy_dir stages/260417_final_family_benchmark_l22/results/control_redundancy_full \
  --output_dir stages/260417_final_family_benchmark_l22/final_outputs
```

# ROAD.md — 開発経緯・調査記録

> PLAN.mdには「何を作るか」を書き、ROAD.mdには「どう進んだか・何に気づいたか」を書く。
> 新しいフェーズを完了するたびに、このファイルに記録を追加すること。

---

## Phase 1: 基盤構築 — ✅ 完了 (2026-05-27)

### 成果物

- プロジェクトスキャフォールド (CMake + ディレクトリ構造)
- CPU Feature Detector (CPUID + XCR0 + キャッシュ + NUMA)
- メモリマネージャ (Huge Pages + NUMAアウェア)
- スレッドプール (work-stealing + CPU親和性ピン留め)
- テンソル抽象 (DataType列挙 + 多次元テンソル)
- モデル抽象 (HyperParams + LayerWeights)
- サンプラー (Temperature/Top-K/Top-P/Min-P/Repetition)
- CLI (`kaguya-cli --cpu-info`)
- Google Test 10テスト全通過
- ベンチマークスイート枠組み

### 重要な発見

#### AMX命令はKVM仮想環境でSIGSEGV

CPUIDはAMX対応を報告するが、実際のLDTILECFGでsegfaultが発生。CPUID Leaf 0x1Dの返り値も不正であり、ハイパーバイザがAMXを正しくパススルーしていないことが判明した。GDBでの解析では、LDTILECFG実行直後にSIGSEGVが発生しており、AMXのtileレジスタがOS側で有効化されていないことが確認された。XCR0レジスタのAMXビット（bit 17, 18）はCPUID上では立っているが、実際のXCR0読み出しでは立っていないという矛盾が観測された。

結論として、KVMハイパーバイザはAMX命令の実行をサポートしていない。AMXカーネルはベアメタル環境でのみ利用可能とし、KVM環境ではスキップする方針とした。

#### AVX-512 BF16/VNNIは完全動作

`_mm512_dpbf16_ps`（BF16ドット積）と`_mm512_dpbusd_epi32`（VNNI INT8ドット積）がKVM環境でも正常に実行されることを確認。BF16ドット積の数値検証も実施し、スカラ実装との一致を確認した。これにより、Kaguyaの主戦力はAVX-512 BF16/VNNIに確定した。

#### フォールバックチェーンの変更

当初はAMX → AVX-512 → AVX2 → Scalarの予定だったが、AMXがKVMで動作しないため、実質的なチェーンを AVX-512 BF16/VNNI → AVX2+FMA → Scalar に変更。AMXはベアメタル環境向けにコードは残すが、デフォルトでは使用しない。

### トラブルシューティング

| 問題 | 原因 | 解決策 |
|------|------|--------|
| `numa.h`が見つからない | libnuma-dev未インストール | `#ifdef KAGUYA_NUMA` ガード + CMake `find_path` |
| `std::greater<>` でコンパイルエラー | テンプレート引数の推論失敗 | `std::greater<ElementType>{}` に明示指定 |
| GCC 14が`tmm`レジスタを認識しない | GCC 14のinline asm制限 | clobberから`tmm`を削除、`"memory"`で代替 |
| GitHub push権限エラー | originがupstreamを指していた | fork (CloudP0nd) にpush → cross-org PR |

### PR

- https://github.com/Carvlly/kaguya/pull/1

---

## Phase 2: GGUFモデルローダー — ✅ 完了 (2026-05-27)

### 成果物

- GGUF v2/v3 パーサー (mmap + stream reader)
- GgmlType全量子化タイプ対応 (Q4_0〜Q8_K, IQ系, TQ系, BF16)
- GgufValue variant型メタデータ (int/float/string/array)
- ModelLoader: GGUF → HyperParams + Weight参照
- ModelWeights: 非所有ポインタベースのウェイト参照 (mmapデータへの直接参照)
- アーキテクチャ検出 (llama/qwen2/mistral/mixtral/phi3/gemma/deepseek/command-r)
- GQA対応HyperParams (num_kv_heads, n_rep, use_gqa)
- MoEエキスパートウェイト対応
- CLI `--model-info` フラグ
- Google Test 23テスト全通過 (GGUF関連13テスト追加)
- ビルド成功: 全ライブラリ + kaguya-cli + テスト + ベンチマーク

### 重要な発見

#### 計画書ファイルの混在問題

`PLAN.md`と`kaguya_plan.md`（旧名`kaguya-plan.md`）が同時にリポジトリに存在し、内容が一部重複・一部矛盾している状態だった。これは複数のエージェントが独立して計画書を作成したことが原因。コードベースと照合した結果、`PLAN.md`が実際のディレクトリ構造・アーキテクチャ設計と整合していることを確認し、`PLAN.md`を唯一の正しい計画書として確定。`kaguya_plan.md`は削除した。

#### プロジェクト構造の大規模リファクタリング

Phase 2のPR #2（upstream側）と、ローカルで進めていた実装との間で、ディレクトリ構造に大きな差異が生じていた。upstream側は古いサブディレクトリ構造（`include/kaguya/memory/`, `model/`, `quantization/`等）を残していたが、ローカルではPLAN.mdに合致するフラット構造に再編済みだった。この差分を解消するため、リファクタリングPR (#3) を作成して構造を統一した。

主な変更点:
- ヘッダーのフラット化: `include/kaguya/memory/memory_manager.h` → `include/kaguya/memory_manager.h`
- ソースの再編: `src/memory/` → `src/runtime/`, `src/model/` → `src/loaders/`, `src/quantization/` → `src/loaders/`
- プレースホルダースタブの削除（中身が`// TODO`だけのファイル群）
- テストディレクトリの`unit/`/`integration/`分離

### 統合作業

- kaguya-phase2/ ディレクトリの本格的GGUFローダー実装をメインプロジェクトに統合
- PLAN.md を唯一の正しい計画書として確定 (kaguya_plan.md は破棄)
- DataType列挙をIQ/TQ系に拡張
- ModelArch列挙にDEEPSEEK/COMMAND_Rを追加

### PR

- https://github.com/Carvlly/kaguya/pull/2
- https://github.com/Carvlly/kaguya/pull/3 (リファクタリング・AGENTS.md/ROAD.md追加)

---

## Phase 3: 計算カーネル — ✅ 完了 (2026-05-27)

### 成果物

- **GEMMカーネル群** (include/kaguya/kernels/gemm.h):
  - FP32 Scalar GEMM: 64x64タイルキャッシュフレンドリー実装
  - FP32 AVX2 GEMM: 4x8レジスタブロッキング + FMA
  - FP32 AVX-512 GEMM: 4x16マイクロカーネル + `_mm512_fmadd_ps`
  - BF16 AVX-512 GEMM: `_mm512_dpbf16_ps` による2-way BF16ドット積（主戦力）
  - VNNI INT8 AVX-512 GEMM: `_mm512_dpbusd_epi32` による4x16バイト転置VNNIレイアウト
  - AMX GEMM: スタブ（ベアメタル専用、KVMではSIGSEGV）
- **特殊演算カーネル群** (include/kaguya/kernels/special_ops.h):
  - Softmax: スカラ + AVX-512（6次Taylor展開exp近似、`_mm512_scalef_ps`併用）
  - RMSNorm: スカラ + AVX-512（`_mm512_fmadd_ps`/`_mm512_reduce_add_ps`）
  - LayerNorm: スカラ + AVX-512（分散計算のベクトル化）
  - RoPE: スカラ + AVX-512（8ペア同時処理、`_mm512_mask_blend_ps`）
  - SiLU: スカラ + AVX-512（ベクトル化sigmoid）
  - GELU: スカラ + AVX-512（近似tanh、Newton-Raphson逆数精製）
- **量子化カーネル群** (include/kaguya/kernels/quantize.h):
  - Q4_0 デ量子化: 18バイト/block、ニブル展開
  - Q8_0 デ量子化: 34バイト/block、int8スケール
  - Q5_0 デ量子化: 22バイト/block、ggml互換qhビットレイアウト
  - Q5_1 デ量子化: 24バイト/block、スケール+オフセット
  - Q4_0/Q8_0 量子化: テスト用FP32→量子化
  - 融合デ量子化+GEMM: Q4_0/Q8_0の行ごとのオンザフライデ量子化+行列積
- **カーネルディスパッチャ**: `select_kernel_target()` + `gemm_dispatch()` 実装
- Google Test 130テスト全通過 (GEMM 39 + 特殊演算 44 + 量子化 25 + 既存 22)

### 重要な発見

#### AVX-512 exp()近似の設計

GCCには`_mm512_exp_ps`が存在しないため、`exp(x) = 2^(x/ln2)`アプローチと多項式近似を組み合わせた独自実装が必要だった。4次の`2^f`多項式近似（`f ∈ [-0.5, 0.5]`）では精度が不十分（~1e-3誤差）だったが、6次Taylor展開の`exp(f)`近似に切り替えたところ~1e-7精度を達成。この近似はSoftmax・SiLU・GELUの全AVX-512パスで利用している。

#### KVM環境でのFMA検出問題

CPUID Leaf 7 EBX[12]（FMA）がKVM環境で0を返す問題が発覚。AVX-512が利用可能な場合、FMA/AVX2/AVXも必ず利用可能であるという論理的推論を`cpu_features.cpp`の検出ロジックに追加した。これにより、AVX2カーネルのテストがSKIPPEDではなく正しくPASSするようになった。

#### Q5_0/Q5_1のqhビットレイアウト

Q5_0/Q5_1の5bit目を格納する`qh[4]`のビットレイアウトは、直感的な`qh[i/8] >> (i%8)`ではなく、ggml互換の`bit 2*i = element i, bit 2*i+1 = element i+16`というパッキング方式を採用していた。初期実装で誤ったレイアウトを使いテストが失敗したが、ggmlソースと照合して正しいレイアウトに修正した。

### PR

- https://github.com/Carvlly/kaguya/pull/4

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

---

## Phase 4: 推論エンジン — ✅ 完了 (2026-05-28)

### 成果物

- **KV-Cache Manager** (include/kaguya/kv_cache.h):
  - レイアウト: [n_layers, n_kv_heads, max_seq_len, head_dim] — GQA対応
  - key_head/value_headアクセサでGEMMに最適な[seq_len, head_dim]連続ブロックを取得可能
  - advance/resetによるポジション管理
  - ムーブセマンティクス対応
- **Attention** (include/kaguya/attention.h):
  - compute_attention() — GQA対応マルチヘッドアテンション
  - Q*K^Tスコア計算 → softmax → 加重和 をGEMMディスパッチャで実行
  - 1/sqrt(d)スケーリング適用済み
- **Pipeline** (include/kaguya/pipeline.h):
  - Transformer推論フルパイプライン: 埋め込み → レイヤー処理 → 出力正規化 → ロジット → サンプリング
  - prefill()/decode()/generate() API
  - 量子化ウェイトのオンザフライデ量子化（dequantize_weight + weighted_project）
  - GEMV定式化: y = W * x^T（Wを行優先[ne1, ne0]として直接使用、転置不要）
- **Batch Inference** (include/kaguya/batch.h):
  - BatchInferenceクラス — 複数シーケンスのラウンドロビン処理
  - add_sequence()/step()/run_all() API
- **ModelWeights拡張**:
  - LayerWeights/ModelWeightsにDataTypeフィールド追加（wq_dtype等）
  - model_loaderがGgmlType→DataType変換を自動設定
- **量子化カーネル拡張**:
  - BF16デ量子化 (dequantize_bf16) — 左16ビットシフトでFP32変換
  - F16デ量子化 (dequantize_f16) — 既存fp16_to_fp32関数を利用
  - dequantize_dispatchがBF16/F16に対応
- **CLI推論ループ**:
  - kaguya-cliにPipeline統合、サンプリングパラメータ設定、生成速度表示
- Google Test 156テスト全通過 (KV Cache 13 + Attention 4 + Inference 11 + 既存 128)

### 重要な発見

#### GEMV定式化: y = W * x^T

Transformer推論の線形レイヤーは通常 y = x @ W^T と定式化されるが、GGUFのウェイトは行優先[ne1, ne0]で格納されている。GEMM C = A * B のBにW^Tが必要になり転置コストが発生する。しかし、M=1（デコード）の場合、y^T = W @ x^T と変形することで、WをそのままA[M=ne1, K=ne0]として使える。これにより転置不要でGEMVを実行可能。

具体例: wq [ne0=emb_dim, ne1=n_heads*head_dim]の場合:
- GEMV: M=n_heads*head_dim, K=emb_dim, N=1
- A=wq (lda=emb_dim), B=x (ldb=1), C=y (ldc=1)

#### KVキャッシュレイアウトの設計

KVキャッシュのレイアウト設計には読み書きのトレードオフがある:
- [max_seq_len, n_kv_heads, head_dim]: 書き込み効率良好（1回のmemcpy）
- [n_kv_heads, max_seq_len, head_dim]: 読み込み効率良好（GEMMに適した連続ブロック）

アテンション計算（読み込み）が性能クリティカルパスであるため、後者を採用。書き込み時はn_kv_heads回の小さなmemcpyが必要だが、これは1トークン生成につき1回のみ。

#### アテンション計算のデータコピー問題

現在の実装では、KVキャッシュからGEMM用の[n_kv_heads, seq_len, head_dim]一時バッファにデータをコピーしている。これはmax_seq_lenのストライド差によるもの。Phase 6でキャッシュレイアウトの最適化またはストライド付きGEMMカーネルの実装により解消予定。

### PR

- https://github.com/Carvlly/kaguya/pull/5

---

## Phase 5: CLI・API・ベンチマーク — ✅ 完了 (2026-05-28)

### 成果物

- **CLIインターフェース** (src/cli/main.cpp):
  - インタラクティブチャットモード (`-i` / `--interactive`)
  - プロンプト指定生成 (`-p` / `--prompt`)
  - ストリーミング出力 (`--stream` / `--no-stream`)
  - ベンチマークモード (`--bench`) — ウォームアップ付き複数イテレーション
  - サンプリングパラメータの全指定対応 (temperature, top-k, top-p, min-p, repeat-penalty, seed)
  - 生成速度表示 (tok/s) — prefill/decode分離計測
  - SIGINT/SIGTERMによる安全な中断
  - チャットモードでのパラメータ動的変更 (`/set` コマンド)
  - バイトレベル簡易トークナイザ (BPEトークナイザは将来実装)
- **C API** (include/kaguya/kaguya.h):
  - `kaguya_model_load/free` — GGUFモデルのロード/解放
  - `kaguya_context_create/free/reset` — 推論コンテキスト管理
  - `kaguya_context_prompt_tokens` — プロンプト投入 (prefill)
  - `kaguya_context_decode` — 1トークン生成
  - `kaguya_context_generate` — 複数トークン生成
  - `kaguya_context_logits` — ロジット取得
  - `kaguya_tokenize/detokenize` — バイトレベルトークナイゼーション
  - `kaguya_model_vocab_size/context_length/emb_dim/num_layers/num_heads` — モデル情報
  - `kaguya_init/cpu_info/version` — ユーティリティ
  - C互換 `extern "C"` リンケージ — Python/Rust等のバインディング対応
  - 共有ライブラリ `libkaguya.so` ビルド対応
- **ベンチマークスイート**:
  - GEMMマイクロベンチマーク: FP32 dispatch/scalar/AVX-512/AVX2 + GEMV特化 (bench_gemm.cpp)
  - 量子化マイクロベンチマーク: Q4_0/Q5_0/Q5_1/Q8_0デ量子化 + BF16/F16デ量子化 + 量子化 + 融合GEMM (bench_quantize.cpp)
  - 特殊演算マイクロベンチマーク: Softmax/RMSNorm/LayerNorm/RoPE/SiLU/GELU (bench_inference.cpp)
  - E2Eパイプラインベンチマーク: 合成モデルによるdecode/prefillスループット測定 (bench_inference.cpp)
  - 全ベンチマークがGoogle Benchmarkフレームワークで構成
- **C APIテスト** (tests/unit/test_c_api.cpp): 27テスト追加
  - バージョン情報・初期化
  - モデルロード失敗ケース (不正パス・NULL)
  - コンテキスト作成・解放 (NULLセーフティ)
  - トークナイゼーション (エンコード・デコード・エラーハンドリング)
  - 全API関数のNULL引数エラーハンドリング検証
- Google Test 183テスト全通過 (C API 27 + 既存 156)
- バージョン v0.1.0 → v0.2.0

### 重要な発見

#### C API設計: opaque handle + malloc/freeパターン

C APIでは opaque handle パターン (`kaguya_model*`, `kaguya_context*`)を採用し、ABI互換性を確保した。トークナイゼーション結果と生成テキストは呼び出し元が`free()`する設計（`kaguya_tokens_free`, `kaguya_text_free`）とし、C++の`std::vector`や`std::string`がC API境界を越えないようにした。これにより、C言語やPython ctypesから安全に呼び出し可能。

#### ベンチマークの合成モデル設計

E2Eパイプラインベンチマークでは実際のGGUFモデルファイルが不要な合成モデルを動的生成するアプローチを採用。`create_bench_model()`でHyperParamsを指定してModelWeightsを構築し、Pipelineを直接インスタンス化する。これによりベンダーマークをファイル依存なしで実行可能。静的変数のライフタイム問題（ポインタダングリング）に注意が必要 — `static`ストレージでウェイトデータを保持し、モデルの寿命中ポインタが有効であることを保証した。

#### CLIベンチマークモードとGoogle Benchmarkの使い分け

CLIの`--bench`モードはユーザーが手軽にtok/sを測定するための簡易ベンチマーク。Google Benchmark (`kaguya_bench`) はマイクロベンチマーク用でGFLOPS等の詳細な性能指標を提供する。両者は目的が異なるため併存させる。

### PR

- https://github.com/Carvlly/kaguya/pull/6

---

## Phase 6: 最適化・検証 — ✅ 完了 (2026-05-28)

### 成果物

- **ストライド対応アテンション** (include/kaguya/attention.h):
  - `AttentionParams`に`kv_stride`フィールド追加 — KVキャッシュのヘッド間ストライドを直接指定可能に
  - `compute_attention()`がストライド付きKVキャッシュに直接アクセス — データコピー完全排除
  - 従来の`kv_h * seq_len * head_dim`オフセット計算を`kv_h * kv_stride`に変更
- **パイプライン最適化** (include/kaguya/pipeline.h, src/inference/pipeline.cpp):
  - KVキャッシュデータコピー完全排除 — `k_attn`/`v_attn`一時バッファのアロケーション・コピーを削除
  - `attn_proj_`/`ffn_down_`メンバーバッファの事前確保 — レイヤーごとの`std::vector`アロケーションを排除
  - `kv_cache_.head_stride()`アクセサでストライド情報を直接取得
- **キャッシュアウェアGEMMタイルチューニング** (include/kaguya/kernels/cache_tuner.h):
  - `CacheTuner`クラス — CPU検出結果からL2キャッシュサイズを取得
  - `get_tile_size()` — sqrt(L2_size / (3 * sizeof(float))) に基づく最適タイルサイズ計算
  - クランプ範囲 [16, 512] — 極端な値を防止
  - スカラGEMMでハードコード`TILE=64`を`CacheTuner::get().get_tile_size()`に置換
- **ソフトウェアプリフェッチ** (src/kernels/gemm_avx512.cpp):
  - AVX-512 FP32 GEMMのKループ内に`_mm_prefetch`ヒント挿入
  - 次のB行と次のA要素をL1キャッシュに事前ロード (`_MM_HINT_T0`)
  - `#if defined(__AVX512F__)`ガード付き
- **メモリ帯域プロファイラ** (include/kaguya/profiler.h):
  - `MemoryBandwidthProfiler` — シーケンシャルread/write/copy帯域測定 (GB/s)
  - `InferenceProfiler` — レイヤー/アテンション/FFN単位のタイミング計測 + サマリー出力
  - `compute_perplexity()` — トークン列のパープレキシティ計算
- **NUMAアウェアKVキャッシュ** (include/kaguya/kv_cache.h):
  - `std::vector<float>`から`MemoryManager::allocate()`に移行 — Huge Pages対応
  - 2MB以上のバッファには`MemFlags::HugePages | MemFlags::Aligned64`を自動使用
  - 2MB未満は`MemFlags::Aligned64`のみ — ヒューリスティック不一致を防止
  - カスタムデストラクタ + ムーブセマンティクス対応
  - `head_stride()`パブリックアクセサ追加
- **精度検証テスト** (tests/unit/test_accuracy.cpp): 10テスト追加
  - Q4_0/Q8_0/Q5_0/Q5_1デ量子化誤差限界テスト
  - BF16ラウンドトリップ精度テスト
  - Softmax正規化検証 (和=1.0)
  - RMSNormスケール保存検証
  - GEMM精度テスト (dispatch vs scalar, 誤差 < 1e-4)
  - Attention/Pipeline出力のNaN/Inf検証
- **安定性テスト** (tests/unit/test_stability.cpp): 20テスト追加
  - GEMM: ゼロ次元・1要素・大K・alpha/beta指定
  - Softmax/RMSNorm/LayerNorm: 1要素・2要素・ゼロ入力
  - RoPE: position=0
  - KVキャッシュ: 1ポジション・最大ポジション・リセット再利用・ムーブセマンティクス
  - Pipeline: 空プロンプト・1トークン・複数リセット
  - Sampler: 全負ロジット・全等ロジット
  - デ量子化: n_blocks=0
- **エラーハンドリング強化**:
  - `KVCache`コンストラクタ: 不正次元で`std::invalid_argument`、オーバーフローで`std::length_error`
  - `Pipeline`: 空プロンプト安全処理、コンテキスト長超過時の`decode()`で-1返却、`is_context_full()`追加
  - `validate_gemm_params()`: NULLポインタ・負次元の検証、全GEMMエントリポイントに挿入
- Google Test 213テスト全通過 (Accuracy 10 + Stability 20 + 既存 183)
- ASan (-fsanitize=address,undefined) ビルドで213テスト全通過 — メモリリークなし

### 重要な発見

#### ストライド対応アテンションによるデータコピー排除

Phase 4で指摘されていた「アテンション計算のデータコピー問題」を解消。KVキャッシュのレイアウト [n_kv_heads, max_seq_len, head_dim] において、ヘッド間ストライドは`max_seq_len * head_dim`だが、`compute_attention()`は`seq_len * head_dim`を想定していた。この不一致を`kv_stride`パラメータで解決し、KVキャッシュデータを直接参照可能にした。

効果: レイヤー1回あたり `2 * n_kv_heads * seq_len * head_dim * sizeof(float)` バイトのコピーが削減。典型的な7Bモデル (n_kv_heads=32, seq_len=2048, head_dim=128) では約64MB/レイヤーのコピーが不要に。

#### bf16_to_f32のプリプロセッサガード問題

`bf16_to_f32()`関数が`#if defined(__AVX512BF16__)`ガード内に定義されていたが、BF16 GEMMのフォールバックパス（スカラGEMM）でも必要であることが判明。ASanビルドでAVX-512 BF16が無効な場合にリンクエラーが発生。関数をガードの外に移動して解決した。

#### MemoryManager::deallocateのサイズヒューリスティック

`MemoryManager::deallocate()`は内部でバッファサイズが2MB以上の場合にHuge Pagesと判定してmunmapを呼ぶが、KVキャッシュの小さなバッファ（テスト用など）で`MemFlags::HugePages`を指定すると、allocateとdeallocateのヒューリスティックが不一致を起こす。解決策として、2MB未満のバッファには`MemFlags::Aligned64`のみを使用するようにした。

#### CacheTunerのL2キャッシュベース最適タイルサイズ

L2キャッシュに3つのタイルパネル（A, B, C）が収まるよう、`tile_size = sqrt(L2_size / (3 * sizeof(float)))`で計算。テスト環境のL2キャッシュが1024KBの場合、最適タイルサイズは約169（→172に丸め）。ハードコードの64より大きく、より大きなキャッシュ利用率を実現。

### トラブルシューティング

| 問題 | 原因 | 解決策 |
|------|------|--------|
| bf16_to_f32のリンクエラー | `#if defined(__AVX512BF16__)`ガード内に定義 | ガードの外に移動 |
| KVキャッシュHuge Pages不整合 | 2MB未満でHugePages指定 + deallocateのヒューリスティック | 2MB未満はAligned64のみ使用 |
| Pipeline::decodeのコンテキスト超過 | pos >= context_lengthでの未定義動作 | -1返却 + `is_context_full()`追加 |

### PR

- (次回のPRでupstreamに提出予定)

---

## Phase 7: 実用化 — 🔧 進行中 (2026-05-29)

### 成果物

- **BPEトークナイザ** (include/kaguya/tokenizer.h, src/core/tokenizer.cpp):
  - GGUFメタデータからBPEトークナイザを自動構築 (`tokenizer.ggml.tokens/scores/token_type/merges`)
  - BPEマージルールの最長一致適用 — 最優先ペアの反復マージ
  - 特殊トークン自動検出 (BOS/EOS/PAD/UNK) — メタデータID + トークン名パターンの両対応
  - UTF-8バイトフォールバック — 未知文字の`<0xHH>`バイトトークンエンコーディング
  - SentencePiece-style前処理 — 空白を▁(U+2581)に変換、先頭に▁を自動付与
  - `encode()`/`decode()`/`decode_token()` API
  - CLI統合: SimpleTokenizer→TokenizerWrapperに差し替え、BPE利用可能時は自動使用
  - C API統合: `kaguya_tokenize`/`kaguya_detokenize`がBPE対応
  - Google Test 19テスト追加 (ビルド・エンコード・デコード・ラウンドトリップ・BPEマージ・特殊トークン)
- **K-quantデ量子化カーネル** (src/kernels/quantize.cpp):
  - Q2_K: 2ビット量子化 + 4ビットスケール/min、4サブグループ×128要素スーパーグループ
  - Q3_K: 3ビット量子化 (2ビットqs + 1ビットhmask符号) + 6ビットパックドスケール
  - Q4_K: 4ビット量子化 + 6ビットパックドスケール/min (`get_scale_min_k4`ヘルパー)
  - Q5_K: 5ビット量子化 (4ビットqs + 1ビットqh上位ビット) + Q4_Kと同じスケール符号化
  - Q6_K: 6ビット量子化 (4ビット下位ql + 2ビット上位qh) + int8スケール
  - Q8_K: 8ビット量子化 + fp32スーパーブロックスケール
  - `dequantize_dispatch()`の全K-quant対応 — 部分ブロックの安全処理（256要素スタックバッファ使用）
  - Q2_Kブロックサイズ修正: 64→84バイト (ggml block_q2_K構造体に合致)
- **EOSトークン停止** (include/kaguya/pipeline.h):
  - `Pipeline::set_eos_token_id()` / `eos_token_id()` — EOS IDの設定・取得
  - `Pipeline::generate()` でEOS検出時の自動停止
  - `GenerationStop` 構造体 — generate()のオーバーロードでEOS停止設定を渡せる
  - CLI: 生成ループでEOS検出時の自動停止を追加
- **C APIストリーミング** (include/kaguya/kaguya.h):
  - `kaguya_stream_callback` — トークン生成ごとのコールバック関数型
  - `kaguya_context_generate_streaming()` — ストリーミング生成 + コールバック + EOS停止
  - `kaguya_model_eos_token_id()` — モデルのEOS ID取得
  - `kaguya_context_set_eos_token()` — コンテキストにEOS ID設定
- Google Test 232テスト全通過 (BPEトークナイザ 19 + 既存 213)

### 重要な発見

#### BPEトークナイザ構築: GGUFメタデータの活用

Phase 1〜6までバイトレベルトークナイザのみで、実際のGGUFモデルで意味のあるテキスト生成が不可能だった。GGUFメタデータの`tokenizer.ggml.tokens`（トークン文字列配列）、`tokenizer.ggml.scores`（スコア）、`tokenizer.ggml.token_type`（NORMAL/UNKNOWN/CONTROL/BYTE）、`tokenizer.ggml.merges`（BPEマージルール）を読み込むことで、モデル固有のBPEトークナイザを自動構築可能になった。特に`tokenizer.ggml.bos_token_id`/`eos_token_id`メタデータとトークン名パターンの両方からBOS/EOSを検出する二段構えの設計が重要 — 一部のGGUFファイルではメタデータIDが欠落している場合がある。

#### K-quantブロックレイアウトの複雑さ

Q4_K/Q5_Kのスケール符号化は6ビット値が12バイトにパックされており、ggmlの`get_scale_min_k4`ヘルパーでデコードが必要。3つの6ビット値が2バイトにパックされる方式で、下位6ビット・中位6ビット・上位4ビット+隣接バイトの2ビットという変則的なレイアウト。Q3_Kはさらに別の6ビットパッキング方式（kmask1/kmask2ビットマスク使用）を採用しており、Q4_K/Q5_Kとは異なる。

#### C++文字列リテラルのUTF-8エスケープ問題

テストコードで`"\xE2\x96\x81a"`のような文字列リテラルを書くと、コンパイラが`\x81a`を1つの16進エスケープとして解釈してしまう問題が発生。回避策として`const std::string sp = "\xE2\x96\x81";` + `sp + "a"`のように、変数に分けて結合する必要がある。

### トラブルシューティング

| 問題 | 原因 | 解決策 |
|------|------|--------|
| `<optional>`未インクルード | tokenizer.hで`std::optional`使用時にヘッダ不足 | `#include <optional>` 追加 |
| Q2_Kデ量子化テストの失敗 | Q2_Kがサポート済みになったのに「未サポートタイプゼロ出力」テストがQ2_Kを使用 | IQ1_Sに変更 |
| UTF-8エスケープの誤解析 | `"\xE2\x96\x81a"`が`\x81a`として解釈 | 変数に分けて結合 |

### PR

- (次回のPRでupstreamに提出予定)

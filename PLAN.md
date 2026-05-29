# Kaguya — CPU推論エンジン 開発計画

> llama.cppを超えるCPU推論エンジン

---

## 1. プロジェクト概要

### 1.1 目標
- llama.cpp比で **1.5x〜2x** の推論スループットを実現するCPU推論エンジン
- GGUFフォーマット互換で既存モデル資産をそのまま利用可能
- AMX / AVX-512 / VNNI などモダンなx86 SIMDを最大限活用

### 1.2 差別化戦略（llama.cppに対する優位性）

| 領域 | llama.cpp | Kaguya |
|------|-----------|--------|
| AMX活用 | 部分的・実験的 | **最適化カーネルの主軸** (※仮想環境では利用不可の場合あり) |
| AVX-512 BF16/VNNI | 部分的 | **実証済み — 実環境で動作確認済** (主戦力) |
| メモリ管理 | mmap中心 | **Huge Pages + NUMAアウェア + カスタムアロケータ** |
| スレッド管理 | スレッドプール | **CPU親和性ピン留め + 非同期パイプライン** |
| 量子化フォーマット | Q4_0〜Q8_0, K-quants | GGUF互換 + **BF16ネイティブ演算 + VNNI INT8** |
| 推論パイプライン | 同期レイヤー逐次 | **レイヤー間プリフェッチ + 非同期実行** |
| キャッシュ最適化 | 限定的 | **L1/L2キャッシュサイズ検出 + タイルサイズ自動調整** |

### 1.3 対応アーキテクチャ
- **プライマリ**: x86_64 (AVX-512 + AMX)
- **セカンダリ**: x86_64 (AVX2のみ、AMXなし環境)
- **将来的**: ARM NEON/SVE (Phase 2)

---

## 2. 技術スタック

| 項目 | 選択 | 理由 |
|------|------|------|
| 言語 | **C++23** | ゼロオーバーヘッド抽象化 + モダン機能 |
| ビルド | **CMake 3.25+** | 広範な対応 + ツールチェーン柔軟性 |
| SIMD内関数 | **インラインアセンブラ + intrinsics** | AMXはintrinsicsが未整備なため |
| スレッド | **C++23 std::jthread + OS API** | 親和性制御にはpthread/Windows API |
| メモリ | **Huge Pages + カスタムアロケータ** | TLBミス削減が推論速度に直結 |
| テスト | **Google Test** | C++デファクトスタンダード |
| ベンチマーク | **Google Benchmark** | マイクロベンチマーク必須 |

---

## 3. アーキテクチャ設計

### 3.1 レイヤー構成

```
┌─────────────────────────────────────────────┐
│              CLI / API Layer                 │  ← ユーザーインターフェース
├─────────────────────────────────────────────┤
│           Inference Engine                   │  ← 推論オーケストレーション
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Pipeline     │  │ KV-Cache Manager     │  │
│  │ Scheduler   │  │ (Ring Buffer)         │  │
│  └─────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────┤
│          Computation Kernels                 │  ← 計算カーネル
│  ┌──────┐ ┌───────┐ ┌──────┐ ┌──────────┐  │
│  │ AMX  │ │AVX512 │ │AVX2  │ │ Fallback │  │
│  │Kernel│ │Kernel │ │Kernel│ │ (Scalar) │  │
│  └──────┘ └───────┘ └──────┘ └──────────┘  │
├─────────────────────────────────────────────┤
│           Runtime Services                   │  ← 基盤サービス
│  ┌──────────┐ ┌─────────┐ ┌─────────────┐  │
│  │ Thread   │ │ Memory  │ │ CPU Feature │  │
│  │ Pool     │ │ Manager │ │ Detector    │  │
│  └──────────┘ └─────────┘ └─────────────┘  │
├─────────────────────────────────────────────┤
│            Model Loader                      │  ← GGUFパーサー
└─────────────────────────────────────────────┘
```

### 3.2 コアモジュール

1. **CPU Feature Detector** — 起動時にCPUIDで対応命令セットを検出
2. **Kernel Dispatcher** — 実行環境に最適なカーネルをランタイム選択
3. **AMX GEMM Kernel** — AMX_TILE/AMX_BF16/AMX_INT8を使ったGEMM
4. **AVX-512 GEMM Kernel** — VNNI/BF16/FP16を使ったGEMM
5. **Quantization Kernels** — デ量子化 + GEMM融合カーネル
6. **KV-Cache Manager** — リングバッファベースのKVキャッシュ
7. **Pipeline Scheduler** — レイヤー間プリフェッチ + 非同期実行
8. **Memory Manager** — Huge Pages + NUMAアウェアアロケータ
9. **Thread Affinity Manager** — CPU親和性ピン留め
10. **GGUF Model Loader** — GGUFフォーマットのパース + メモリマップ

---

## 4. 開発フェーズ

### Phase 1: 基盤構築 (Day 1-3)

#### Task 1.1: プロジェクトスキャフォールド
- CMakeプロジェクト構成
- ディレクトリ構造 (`src/core/`, `src/kernels/`, `src/runtime/`, `include/`, `tests/`, `benchmarks/`)
- CI設定 (将来的)

#### Task 1.2: CPU Feature Detector
- CPUID命令による特徴検出 (AVX, AVX2, AVX-512各種, AMX, FMA, VNNI)
- OS側のXSAVE/YMM/ZMM許可確認
- キャッシュ階層検出 (L1/L2/L3サイズ)
- NUMAトポロジー検出

#### Task 1.3: メモリマネージャ
- Huge Pages (2MB/1GB) アロケーション
- NUMAアウェアメモリ配置
- アラインメント保証 (64バイト境界)
- メモリプール（推論中の動的アロケーション排除）

#### Task 1.4: スレッド管理
- スレッドプール (work-stealing)
- CPU親和性ピン留め (pthread_setaffinity_np)
- ハイブリッド構成対応 (P-core / E-core検出)

---

### Phase 2: GGUFモデルローダー (Day 3-5)

#### Task 2.1: GGUFフォーマットパーサー
- ヘッダー読み込み (マジック, バージョン, テンソル数, メタデータ数)
- メタデータ読み込み (全型対応: uint8〜float64, string, array)
- テンソル記述子読み込み (名前, 型, 次元, オフセット)

#### Task 2.2: テンソルデータローダー
- mmapによる遅延読み込み
- 量子化タイプの列挙定義 (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, BF16, FP16, etc.)
- メモリマップされたテンソルへのアクセサ

#### Task 2.3: モデル構造抽象化
- Transformerアーキテクチャの抽象表現
- LLaMA/Qwen/Mistral等の主要アーキテクチャ対応
- レイヤー単位のテンソル参照

---

### Phase 3: 計算カーネル — 最重要フェーズ (Day 5-10)

#### Task 3.1: AMX GEMMカーネル (★最優先)
- AMX TILE設定 (TILECFG)
- AMX_BF16行列積 (TDPBF16PS)
- AMX_INT8行列積 (TDPBSSD / TDPBSUD)
- タイルサイズ最適化 (キャッシュに収まるサイズの自動選択)
- **目標: llama.cppのAVX-512カーネル比 1.5x〜2x**

#### Task 3.2: AVX-512 GEMMカーネル
- AVX-512F + FMA スループット最適化
- AVX-512_VNNI INT8ドット積
- AVX-512_BF16 BF16ドット積
- AVX-512_FP16 FP16ドット積
- タイル化 + レジスタブロッキング

#### Task 3.3: 量子化カーネル
- デ量子化 + GEMM融合 (メモリ帯域削減)
- Q4_0/Q5_0/Q8_0 → FP32/BF16 融合GEMM
- K-quants → FP32/BF16 融合GEMM
- インラインデ量子化によるキャッシュ効率向上

#### Task 3.4: 特殊演算カーネル
- Softmax (AVX-512最適化)
- RMSNorm / LayerNorm (AVX-512最適化)
- RoPE (回転位置エンコーディング, AVX-512最適化)
- SiLU / GELU 活性化関数
- Attentionスコア計算 (マスク付き)

#### Task 3.5: カーネルディスパッチャ
- 実行時CPU特徴検出に基づく最適カーネル選択
- フォールバックチェーン (AMX → AVX-512 → AVX2 → Scalar)
- マイクロベンチマークによる自動チューニング

---

### Phase 4: 推論エンジン (Day 10-14)

#### Task 4.1: KV-Cache Manager
- リングバッファ実装 (固定長シーケンス)
- Paged-Attention風セグメント管理 (マルチシーケンス対応)
- BF16/FP8 KVキャッシュ (メモリ削減)

#### Task 4.2: Transformer推論パイプライン
- Pre-fill (プロンプト処理) — GEMV最適化
- Decode (トークン生成) — GEMM最適化
- レイヤー間非同期プリフェッチ
- GEMM → 活性化 → GEMM のオーバーラップ実行

#### Task 4.3: サンプリング
- Temperature / Top-K / Top-P / Min-P
- Repetition penalty
- Grammar-constrained sampling (将来的)

#### Task 4.4: バッチ推論
- 複数シーケンスのバッチGEMM
- 動的バッチサイズ調整

---

### Phase 5: CLI・API・ベンチマーク (Day 14-17)

#### Task 5.1: CLIインターフェース
- `kaguya-cli` — インタラクティブ推論
- モデルロードオプション
- 生成パラメータ設定

#### Task 5.2: C API
- `kaguya.h` — シンプルなC API
- モデルロード / コンテキスト作成 / 推論 / 解放

#### Task 5.3: ベンチマークスイート
- マイクロベンチマーク (GEMM, 量子化, 特殊演算)
- エンドツーエンドベンチマーク (tokens/s)
- llama.cppとの比較ベンチマーク

---

### Phase 6: 最適化・検証 (Day 17-21)

#### Task 6.1: キャッシュ最適化
- L1/L2キャッシュに収まるタイルサイズの自動チューニング
- プリフェッチ命令の戦略的挿入
- データレイアウト最適化 (AoS → SoA)

#### Task 6.2: メモリ帯域最適化
- メモリ帯域プロファイリング
- NUMA局所性の検証と最適化
- Huge Pages効果の測定

#### Task 6.3: 精度検証
- llama.cpp出力との一致検証
- パープレキシティ比較
- 量子化精度の回帰テスト

#### Task 6.4: 安定性・エラーハンドリング
- 境界ケースのテスト
- メモリリーク検査 (ASan/Valgrind)
- 異常系のエラーハンドリング

---

## 5. ディレクトリ構造

```
kaguya/
├── CMakeLists.txt
├── AGENTS.md
├── PLAN.md
├── ROAD.md
│
├── include/
│   └── kaguya/
│       ├── kaguya.h              # 公開C API
│       ├── cpu_features.h        # CPU特徴検出API
│       ├── tensor.h              # テンソル抽象
│       ├── model.h               # モデル抽象
│       ├── context.h             # 推論コンテキスト
│       └── sampling.h            # サンプリングAPI
│
├── src/
│   ├── core/
│   │   ├── cpu_features.cpp      # CPUID実装
│   │   ├── tensor.cpp            # テンソル実装
│   │   ├── model.cpp             # モデル管理
│   │   └── context.cpp           # 推論コンテキスト
│   │
│   ├── loaders/
│   │   ├── gguf_parser.cpp       # GGUFパーサー
│   │   └── gguf_tensor.cpp       # テンソルデータアクセス
│   │
│   ├── kernels/
│   │   ├── dispatcher.cpp        # カーネルディスパッチ
│   │   ├── gemm_amx.cpp          # AMX GEMM
│   │   ├── gemm_avx512.cpp       # AVX-512 GEMM
│   │   ├── gemm_avx2.cpp         # AVX2 GEMM
│   │   ├── gemm_scalar.cpp       # スカラフォールバック
│   │   ├── quantize.cpp          # 量子化・デ量子化
│   │   ├── softmax.cpp           # Softmax
│   │   ├── norm.cpp              # RMSNorm/LayerNorm
│   │   ├── rope.cpp              # RoPE
│   │   └── activation.cpp        # SiLU/GELU
│   │
│   ├── runtime/
│   │   ├── memory_manager.cpp    # Huge Pages + NUMA
│   │   ├── thread_pool.cpp       # スレッドプール
│   │   └── thread_affinity.cpp   # CPU親和性
│   │
│   ├── inference/
│   │   ├── pipeline.cpp          # 推論パイプライン
│   │   ├── kv_cache.cpp          # KVキャッシュ
│   │   ├── attention.cpp         # アテンション計算
│   │   └── batch.cpp             # バッチ推論
│   │
│   ├── sampling/
│   │   ├── sampler.cpp           # サンプラー
│   │   └── grammar.cpp           # グラマーサンプリング
│   │
│   └── cli/
│       └── main.cpp              # CLIエントリポイント
│
├── tests/
│   ├── unit/
│   │   ├── test_cpu_features.cpp
│   │   ├── test_gguf_parser.cpp
│   │   ├── test_gemm.cpp
│   │   ├── test_quantize.cpp
│   │   └── test_kv_cache.cpp
│   └── integration/
│       └── test_inference.cpp
│
├── benchmarks/
│   ├── bench_gemm.cpp
│   ├── bench_quantize.cpp
│   └── bench_inference.cpp
│
└── tools/
    ├── compare_with_llama_cpp.py # llama.cpp比較スクリプト
    └── download_model.sh         # テストモデルダウンロード
```

---

## 6. 性能目標

### 6.1 マイクロベンチマーク目標

| カーネル | llama.cpp (AVX-512) | Kaguya目標 (AMX) | 倍率 |
|----------|---------------------|-------------------|------|
| BF16 GEMM (4096×4096) | ベースライン | 1.5x〜2.0x | — |
| INT8 GEMM (Q8_0) | ベースライン | 2.0x〜3.0x | — |
| Q4_0 デ量子化+GEMM融合 | ベースライン | 1.3x〜1.5x | — |

### 6.2 エンドツーエンド目標 (LLaMA-7B Q4_0, 4スレッド)

| メトリクス | llama.cpp | Kaguya目標 |
|------------|-----------|------------|
| Prefill (tokens/s) | ベースライン | 1.3x〜1.5x |
| Decode (tokens/s) | ベースライン | 1.5x〜2.0x |
| 初回レイテンシ | ベースライン | 0.8x以下 |

---

## 7. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| AMXが仮想環境で制限される | 中 | 高 | AVX-512 VNNI/BF16を強力なセカンダリ戦略に |
| Huge Pagesが利用不可 | 低 | 中 | フォールバック (mmap + MADV_HUGEPAGE) |
| GGUF互換性の微妙な差異 | 中 | 中 | テストスイートで回帰検出 |
| スレッド親和性がコンテナで制限 | 中 | 低 | 自動検出 + グレースフルデグラデーション |
| キャッシュサイズ推定のミス | 低 | 中 | 実測ベースの自動チューニング |

---

## 8. マイルストーン

| マイルストーン | 目安 | 成果物 |
|----------------|------|--------|
| **M1: 基盤完成** | Day 3 | CPU検出 + メモリ管理 + スレッド管理 |
| **M2: モデルロード** | Day 5 | GGUFパーサー + テンソルアクセス |
| **M3: カーネル完成** | Day 10 | AMX/AVX-512/量子化カーネル + マイクロベンチ |
| **M4: 推論動作** | Day 14 | End-to-end推論 + KVキャッシュ |
| **M5: リリース候補** | Day 17 | CLI + C API + ベンチマーク |
| **M6: 最適化完了** | Day 21 | llama.cpp比較で目標性能達成 |

---

## 9. 開発状況

開発の経緯・調査記録・重要な発見は **ROAD.md** に記録している。

| フェーズ | 状態 |
|----------|------|
| Phase 1: 基盤構築 | ✅ 完了 |
| Phase 2: GGUFモデルローダー | ✅ 完了 |
| Phase 3: 計算カーネル | ✅ 完了 |
| Phase 4: 推論エンジン | ✅ 完了 |
| Phase 5: CLI・API・ベンチマーク | ✅ 完了 |
| Phase 6: 最適化・検証 | ✅ 完了 |
| Phase 7: 実用化 | 🔧 進行中 | BPEトークナイザ・K-quant対応・EOS停止・並列Prefill・ストリーミングAPI |

---

### Phase 7: 実用化 — リアルモデル推論対応

Phase 1〜6で計算基盤は完成したが、実際のGGUFモデルで推論を行うための必須機能が欠けている。
Phase 7では「合成モデルテストが通る」状態から「実モデルで正しく推論できる」状態への移行を目指す。

#### Task 7.1: BPEトークナイザ (★最優先)
- GGUFメタデータ (`tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, `tokenizer.ggml.token_type`, `tokenizer.ggml.merges`) からBPEトークナイザを構築
- BPEマージルールの適用 (最長一致ペアの反復マージ)
- 特殊トークン対応 (BOS/EOS/PAD/UNK)
- UTF-8バイトフォールバック (未知文字のバイトエンコーディング)
- `include/kaguya/tokenizer.h` — BpeTokenizer クラス
- `src/core/tokenizer.cpp` — 実装
- CLI・C APIへの統合 (SimpleTokenizer差し替え)
- **現在バイトレベルのみ → 実モデルで意味のあるテキスト生成が不可能な状態を解消**

#### Task 7.2: K-quantデ量子化カーネル
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K のブロックレイアウト定義とデ量子化実装
- K-quant融合GEMM (Q4_K, Q5_K, Q6_K の行ごとオンザフライデ量子化+行列積)
- `dequantize_dispatch()` のK-quant対応拡張
- **Q4_K_M/Q5_K_M等の最普及フォーマットで現在ゴミ出力になる問題を解消**

#### Task 7.3: EOSトークン停止 + アーキテクチャ固有フォワードパス
- EOSトークン検出による自動停止 (`Pipeline::decode()` / `Pipeline::generate()`)
- BOSトークン自動付与オプション
- アーキテクチャ固有処理:
  - Gemma: 最終層のRMSNormゲイン乗算差異
  - Qwen2: RMSNormイプシロン値の差異
  - Phi-3: GeLU→SiLU切り替え等
- `Pipeline::generate()` のmax_tokens + EOS停止の両対応

#### Task 7.4: 並列Prefill
- プロンプト処理のバッチGEMM化 (1トークンずつ→シーケンス一括処理)
- Prefill時のM>1 GEMM (Q*K^T をバッチ行列積で一括計算)
- `Pipeline::prefill()` の最適化 — 逐次ループ→バッチGEMV

#### Task 7.5: C APIストリーミングコールバック
- `kaguya_context_generate_streaming(ctx, callback, user_data)` — トークン生成ごとのコールバック
- `kaguya_context_decode_with_text(ctx, ...)` — トークンID + テキスト同時返却
- BPEトークナイザ統合による `kaguya_tokenize` / `kaguya_detokenize` のBPE対応化

#### Task 7.6: 実モデルテストインフラ
- 小規模GGUFモデルダウンロードスクリプト (`tools/download_test_model.sh`)
- E2E推論テスト (実際のGGUFモデルで推論結果の整合性検証)
- BPEトークナイザの正確性テスト (既知テキストのエンコード/デコード往復)
- K-quantデ量子化精度テスト

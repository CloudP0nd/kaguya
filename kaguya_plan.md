# 🌙 Kaguya — CPU超高速LLM推論エンジン

> **Vision**: llama.cppを凌駕する次世代CPU推論エンジン。AVX-512 VNNI/BF16/FP16を最大限に活用し、中小規模LLMのCPU推論において圧倒的パフォーマンスを実現する。

**リポジトリ**: https://github.com/Carvlly/kaguya
**ライセンス**: MIT
**ステータス**: 🚧 開発初期 (Phase 0)

---

## 📊 全体進捗ダッシュボード

| Phase | 名前 | 進捗 | 詳細 |
|-------|------|------|------|
| Phase 0 | プロジェクト基盤 | 🟡 進行中 | 詳細 |
| Phase 1 | コア推論カーネル | ⬜ 未着手 | 詳細 |
| Phase 2 | モデルローダー | ⬜ 未着手 | 詳細 |
| Phase 3 | サンプリング・デコーディング | ⬜ 未着手 | 詳細 |
| Phase 4 | メモリ管理 | ⬜ 未着手 | 詳細 |
| Phase 5 | 高度な最適化 | ⬜ 未着手 | 詳細 |
| Phase 6 | CLI & API | ⬜ 未着手 | 詳細 |
| Phase 7 | ベンチマーク & 検証 | ⬜ 未着手 | 詳細 |

---

## 🏗️ アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kaguya CLI / API                         │
│                    (src/main.cpp, libkaguya)                    │
├─────────────────────────────────────────────────────────────────┤
│                    Sampling Layer                                │
│    top_k / top_p / temperature / rep_penalty / mirostat        │
├─────────────────────────────────────────────────────────────────┤
│              Transformer Inference Engine                       │
│    Attention / FFN / RMSNorm / RoPE / LayerNorm                │
├─────────────────────────────────────────────────────────────────┤
│              KV Cache & Memory Manager                          │
│    PagedAttention / Buffer Pool / Memory-mapped I/O            │
├─────────────────────────────────────────────────────────────────┤
│          Quantization & Format Layer                            │
│    GGUF Loader / INT4 / INT8 / BF16 dequant kernels            │
├─────────────────────────────────────────────────────────────────┤
│                  SIMD Compute Kernels                           │
│    AVX-512 VNNI / BF16 / FP16 / AVX2 fallback / auto-detect   │
├─────────────────────────────────────────────────────────────────┤
│                    Core Runtime                                 │
│    Tensor / Context / Engine / Thread Pool / Timer             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 プロジェクト構造

```
kaguya/
├── CMakeLists.txt              # ルートビルド設定
├── cmake/
│   └── DetectSIMD.cmake        # SIMD自動検出モジュール
├── include/kaguya/
│   ├── common.h                # 共通ユーティリティ・型定義
│   ├── core/
│   │   ├── tensor.h            # テンソルデータ構造
│   │   ├── engine.h            # 推論エンジンインターフェース
│   │   └── context.h           # 推論コンテキスト (GPU的役割)
│   ├── kernels/
│   │   ├── transformer.h       # Transformer全体の推論パイプライン
│   │   ├── attention.h         # Multi-Head Attentionカーネル
│   │   ├── ffn.h               # Feed-Forward Networkカーネル
│   │   ├── rmsnorm.h           # RMSNormカーネル
│   │   ├── rope.h              # Rotary Position Embedding
│   │   └── mul_mat.h           # 行列乗算 (量子化対応)
│   ├── quantization/
│   │   ├── gguf_loader.h       # GGUFファイルフォーマットローダー
│   │   └── quantizer.h         # 量子化/逆量子化ユーティリティ
│   ├── sampling/
│   │   └── sampler.h           # トークンサンプリング
│   ├── memory/
│   │   ├── buf_manager.h       # バッファプールマネージャー
│   │   └── kv_cache.h          # KVキャッシュ管理
│   ├── model/
│   │   ├── loader.h            # モデルローダー
│   │   ├── hparams.h           # ハイパーパラメータ
│   │   └── weights.h           # 重みデータ構造
│   └── utils/
│       ├── logging.h           # ロギングユーティリティ
│       └── timer.h             # 高精度タイマー
├── src/                        # 実装ファイル
├── tests/                      # ユニットテスト
├── benchmarks/                 # パフォーマンスベンチマーク
├── scripts/                    # ビルド・評価スクリプト
├── docs/                       # ドキュメント
├── examples/                   # 使用例
└── third_party/                # サードパーティライブラリ
```

---

## 🗓️ Phase別実装計画

### Phase 0: プロジェクト基盤 (現在)

**目標**: 開発環境の確立とビルドシステムの構築。すべてのチェックポイントが通ればPhase 1へ移行可能。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-0.1 | Gitリポジトリ初期化・ディレクトリ構造作成 | ✅ 完了 | #1 |
| CP-0.2 | CMakeビルドシステムの構築 (SIMD自動検出含む) | ✅ 完了 | #1 |
| CP-0.3 | Tensor データ構造の定義と実装 | ✅ 完了 | #1 |
| CP-0.4 | `kaguya_plan.md` 計画書の作成と進捗管理システム | ✅ 完了 | #1 |
| CP-0.5 | CIワークフロー (GitHub Actions) の設定 | ⬜ 未着手 | — |
| CP-0.6 | 基本のHello Worldビルド動作確認 | ⬜ 未着手 | — |

**完了条件**: `cmake -B build && cmake --build build` が正常に完了すること。

---

### Phase 1: コア推論カーネル

**目標**: llama.cppのコア計算カーネルと同等以上の機能を実装。AVX-512 VNNIの恩恵を最大限に受けるINT8/INT4カーネルを提供する。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-1.1 | AVX-512/AVX2 SIMDディスパッチ機構 | ⬜ 未着手 | — |
| CP-1.2 | RMSNorm カーネル (AVX-512最適化) | ⬜ 未着手 | — |
| CP-1.3 | RoPE (Rotary Position Embedding) カーネル | ⬜ 未着手 | — |
| CP-1.4 | 量子化行列乗算 (mul_mat) — INT8/INT4/FP16/BF16 | ⬜ 未着手 | — |
| CP-1.5 | AVX-512 VNNI専用INT8 dot-productカーネル | ⬜ 未着手 | — |
| CP-1.6 | AVX-512 BF16カーネル (直接演算) | ⬜ 未着手 | — |
| CP-1.7 | Multi-Head Attention カーネル (QKV投影 + 確率計算) | ⬜ 未着手 | — |
| CP-1.8 | Feed-Forward Network (SwiGLU/MoE) カーネル | ⬜ 未着手 | — |
| CP-1.9 | Transformerレイヤー統合パイプライン | ⬜ 未着手 | — |
| CP-1.10 | マルチスレッド推論 (スレッドプール + タスク分割) | ⬜ 未着手 | — |
| CP-1.11 | カーネル単体テスト (正確性検証) | ⬜ 未着手 | — |

**完了条件**: すべてのカーネルテストがPASSし、単体ベンチマークでllama.cppのAVX-512ビルドと同等以上のスループットを確認。

** llama.cpp との差別化戦略 (Phase 1)**:
- **AVX-512 VNNI**: INT8 dot-productを `_mm512_dpbusd_epi32` で1命令実行。llama.cppはAVX2 + emulated INT8に依存する場合が多く、512bit幅×VNNIで理論上4倍のスループット。
- **AVX-512 BF16**: `_mm512_dpbf16_ps` で直接BF16行列乗算。FP16変換のオーバーヘッドを削減。
- **AVX-512 FP16**: `_mm512_mul_ph` / `_mm512_add_ph` でFP16直接演算。将来のFP16モデル対応を見据える。
- **Cache-aware tiling**: L1/L2/L3キャッシュサイズに基づく最適なタイル分割でキャッシュミスを最小化。

---

### Phase 2: モデルローダー

**目標**: GGUFフォーマットの完全な読み込みサポート。既存の量子化モデルをそのまま利用可能にする。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-2.1 | GGUFフォーマットパーサー (ヘッダー + メタデータ) | ⬜ 未着手 | — |
| CP-2.2 | テンソルデータの読み込み (メモリマップI/O対応) | ⬜ 未着手 | — |
| CP-2.3 | 量子化タイプの自動判定と重みデコーディング | ⬜ 未着手 | — |
| CP-2.4 | モデルアーキテクチャの自動判定 (LLaMA/Qwen/Gemma/Mistral他) | ⬜ 未着手 | — |
| CP-2.5 | ハイパーパラメータの読み込みと検証 | ⬜ 未着手 | — |
| CP-2.6 | サポート対象モデルの動作確認 (Qwen2.5-0.5B等) | ⬜ 未着手 | — |

**完了条件**: Hugging Face GGUFモデルをダウンロードしてテキスト生成が可能なこと。

---

### Phase 3: サンプリング & デコーディング

**目標**: 高品質なテキスト生成のためのサンプリング層の実装。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-3.1 | 基本サンプリング (greedy / temperature / top_k / top_p) | ⬜ 未着手 | — |
| CP-3.2 | min_p サンプリング | ⬜ 未着手 | — |
| CP-3.3 | Mirostat v1/v2 サンプリング | ⬜ 未着手 | — |
| CP-3.4 | Repetition penalty / frequency penalty | ⬜ 未着手 | — |
| CP-3.5 | Logit bias サポート | ⬜ 未着手 | — |
| CP-3.6 | Grammar-constrained decoding (json-schema等) | ⬜ 未着手 | — |
| CP-3.7 | 確定的サンプリング (seeded RNG) | ⬜ 未着手 | — |

**完了条件**: llama.cppのサンプリングオプションと完全な互換性を持ち、品質テストで同等以上の結果。

---

### Phase 4: メモリ管理

**目標**: 効率的なKVキャッシュとメモリ管理で、より長いコンテキストを少ないメモリで実現。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-4.1 | バッファプールマネージャー (slab allocator) | ⬜ 未着手 | — |
| CP-4.2 | KVキャッシュの基本実装 (連続メモリ) | ⬜ 未着手 | — |
| CP-4.3 | Paged KV Cache (ページ単位の動的割り当て) | ⬜ 未着手 | — |
| CP-4.4 | KVキャッシュの量子化 (INT8 KV cache) | ⬜ 未着手 | — |
| CP-4.5 | メモリマップI/Oによる巨大モデルの遅延読み込み | ⬜ 未着手 | — |
| CP-4.6 | メモリ使用量の最適化 (shared tensor等) | ⬜ 未着手 | — |

**完了条件**: llama.cppのKVキャッシュメモリ使用量に対して30%以上の削減。

---

### Phase 5: 高度な最適化

**目標**: アーキテクチャレベルの最適化でllama.cppに決定的なリードを築く。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-5.1 | Continuous batching の実装 | ⬜ 未着手 | — |
| CP-5.2 | Speculative decoding (draft model による推論高速化) | ⬜ 未着手 | — |
| CP-5.3 | GQA (Grouped-Query Attention) 最適化 | ⬜ 未着手 | — |
| CP-5.4 | Sliding Window Attention 対応 | ⬜ 未着手 | — |
| CP-5.5 | Prefix caching (共有プロンプトのキャッシュ再利用) | ⬜ 未着手 | — |
| CP-5.6 | NUMA-aware メモリ割り当て (マルチソケット最適化) | ⬜ 未着手 | — |
| CP-5.7 | Graph optimization (オペレーション融合) | ⬜ 未着手 | — |

**完了条件**: 複数リクエストの同時処理でllama.cppの2倍以上のスループット。

---

### Phase 6: CLI & API

**目標**: ユーザーが簡単に使えるコマンドラインツールとライブラリAPIの提供。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-6.1 | インタラクティブCLI (llama-cli互換のオプション) | ⬜ 未着手 | — |
| CP-6.2 | サーバーモード (OpenAI互換 HTTP API) | ⬜ 未着手 | — |
| CP-6.3 | Python バインディング (pybind11) | ⬜ 未着手 | — |
| CP-6.4 | ストリーミング出力 (SSE対応) | ⬜ 未着手 | — |
| CP-6.5 | 埋め込み生成モード (embedding extraction) | ⬜ 未着手 | — |
| CP-6.6 | 設定ファイル読み込み (YAML/TOML) | ⬜ 未着手 | — |

**完了条件**: `kaguya-cli -m model.gguf -p "Hello"` で対話が可能。

---

### Phase 7: ベンチマーク & 検証

**目標**: 客観的なパフォーマンス比較でllama.cppを超えることを証明。

| チェックポイント | 内容 | ステータス | PR |
|-----------------|------|-----------|-----|
| CP-7.1 | 自動ベンチマークフレームワークの構築 | ⬜ 未着手 | — |
| CP-7.2 | Prompt processing (prefill) 速度比較 | ⬜ 未着手 | — |
| CP-7.3 | Token generation (decode) 速度比較 | ⬜ 未着手 | — |
| CP-7.4 | メモリ使用量比較 | ⬜ 未着手 | — |
| CP-7.5 | マルチスレッドスケーリングの測定 | ⬜ 未着手 | — |
| CP-7.6 | 品質テスト (perplexity / 生成品質の比較) | ⬜ 未着手 | — |
| CP-7.7 | 公開ベンチマーク結果の作成 | ⬜ 未着手 | — |

**完了条件**: 複数モデル・複数量子化タイプでllama.cppを10%以上上回るトークン/秒。

---

## 🎯 llama.cppを超えるための技術戦略

### 1. AVX-512 VNNI — 最大の武器

llama.cppは幅広いCPUアーキテクチャに対応するため、AVX2レベルでの最適化が中心。KaguyaはAVX-512 VNNI/BF16/FP16を**前提**とすることで、不可能な最適化を実現：

| 演算 | llama.cpp (AVX2) | Kaguya (AVX-512 VNNI) | 理論速度比 |
|------|-------------------|------------------------|-----------|
| INT8 dot-product | emulated (AVX2) | `_mm512_dpbusd_epi32` | ~4x |
| BF16 matmul | BF16→FP32変換後 | `_mm512_dpbf16_ps` | ~2x |
| INT4 dequant+mul | スカラ拡張 | AVX-512 VBMI2 shuffle | ~2x |
| FP16 add | AVX2 emulated | `_mm512_add_ph` | ~2x |

### 2. キャッシュ最適化

- **L1-aware blocking**: 32KB L1dキャッシュに収まるタイルサイズでループ構築
- **L2 prefetching**: `_mm_prefetch` によるL2キャッシュへの事前フェッチ
- **KVキャッシュのページング**: 使用中のキャッシュのみL3に保持

### 3. スレッドスケーリング

- **Work-stealing thread pool**: 不均衡な負荷を動的に再分配
- **NUMA-aware scheduling**: マルチソケット環境でのローカルメモリ優先アクセス
- **Fine-grained locking**: グローバルロックなしの並行KVキャッシュアクセス

---

## 🔧 開発環境

| 項目 | バージョン |
|------|-----------|
| OS | Linux x86_64 (5.10) |
| CPU | Intel Xeon (4C/4T, AVX-512 VNNI/BF16/FP16) |
| RAM | 8 GB |
| Compiler | GCC 14.2.0 (C++20) |
| CMake | 4.3.2 |
| Ninja | 1.13.0 |
| Python | 3.12.13 |

---

## 📝 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-05-25 | プロジェクト初期化。Phase 0 の基盤構築開始。 |


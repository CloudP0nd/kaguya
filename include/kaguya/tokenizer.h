#pragma once
/// @file tokenizer.h
/// @brief BPE tokenizer for GGUF models.
///
/// Reads tokenizer metadata from GGUF files (tokens, merges, scores, types)
/// and implements Byte Pair Encoding for text tokenization.
/// Supports:
///   - BPE merge rules (longest-match priority)
///   - Special tokens (BOS, EOS, PAD, UNK)
///   - UTF-8 byte fallback for unknown characters
///   - SentencePiece-style preprocessing (▁ space encoding)

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <cstdint>

namespace kaguya {

/// Token type flags (matches GGUF tokenizer.ggml.token_type values)
enum class TokenType : int32_t {
    NORMAL      = 0,
    UNKNOWN     = 1,
    CONTROL     = 2,
    BYTE        = 3,
    UNUSED      = 4,
};

/// BPE Tokenizer — reads from GGUF metadata and implements BPE encoding
class BpeTokenizer {
public:
    BpeTokenizer() = default;

    /// Build tokenizer from GGUF metadata
    /// @param tokens Array of token strings (tokenizer.ggml.tokens)
    /// @param scores Array of token scores (tokenizer.ggml.scores)
    /// @param token_types Array of token types (tokenizer.ggml.token_type)
    /// @param merges Array of merge rules as "A B" strings (tokenizer.ggml.merges)
    /// @param bos_id BOS token ID (from tokenizer.ggml.bos_token_id, or auto-detect)
    /// @param eos_id EOS token ID (from tokenizer.ggml.eos_token_id, or auto-detect)
    /// @param add_bos Whether to automatically prepend BOS token
    /// @return true on success
    bool build(const std::vector<std::string>& tokens,
               const std::vector<float>& scores,
               const std::vector<int32_t>& token_types,
               const std::vector<std::string>& merges,
               int32_t bos_id = -1,
               int32_t eos_id = -1,
               bool add_bos = true);

    /// Check if tokenizer has been built successfully
    bool is_valid() const { return !id_to_token_.empty(); }

    /// Encode text to token IDs
    /// @param text Input text (UTF-8)
    /// @param add_bos Override: prepend BOS token (default: use build-time setting)
    /// @param add_eos Append EOS token
    /// @return Vector of token IDs
    std::vector<int32_t> encode(const std::string& text,
                                 std::optional<bool> add_bos = std::nullopt,
                                 bool add_eos = false) const;

    /// Decode token IDs back to text
    /// @param tokens Array of token IDs
    /// @param skip_special Skip special/control tokens in output
    /// @return Decoded text string (UTF-8)
    std::string decode(const std::vector<int32_t>& tokens,
                        bool skip_special = true) const;

    /// Decode a single token ID to text
    std::string decode_token(int32_t token_id, bool skip_special = true) const;

    /// Get vocabulary size
    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }

    /// Get BOS token ID (-1 if not set)
    int32_t bos_token_id() const { return bos_id_; }

    /// Get EOS token ID (-1 if not set)
    int32_t eos_token_id() const { return eos_id_; }

    /// Get PAD token ID (-1 if not set)
    int32_t pad_token_id() const { return pad_id_; }

    /// Get UNK token ID (-1 if not set)
    int32_t unk_token_id() const { return unk_id_; }

    /// Whether to automatically add BOS
    bool add_bos() const { return add_bos_; }

    /// Get token text by ID
    const std::string& token_text(int32_t id) const;

    /// Check if a token ID is a special token (BOS, EOS, PAD, UNK)
    bool is_special_token(int32_t id) const;

private:
    /// Apply BPE merges to a list of tokens
    /// Returns the merged token sequence
    std::vector<int32_t> apply_bpe(const std::vector<int32_t>& initial_tokens) const;

    /// Convert a single UTF-8 character to its byte-fallback token representation
    /// Returns the byte tokens for a character that has no direct token mapping
    std::vector<int32_t> char_to_byte_tokens(uint32_t codepoint) const;

    /// Preprocess text: SentencePiece-style space replacement
    std::string preprocess_text(const std::string& text) const;

    /// Split text into initial token candidates (character-level + byte fallback)
    std::vector<int32_t> text_to_initial_tokens(const std::string& text) const;

    // Token storage
    std::vector<std::string> id_to_token_;              // id -> token text
    std::vector<float> id_to_score_;                     // id -> score
    std::vector<TokenType> id_to_type_;                  // id -> token type
    std::unordered_map<std::string, int32_t> token_to_id_; // token text -> id

    // BPE merge rules: pair -> merge priority (lower = higher priority)
    // A "pair" is represented as "token_a token_b" (space-separated)
    std::unordered_map<std::string, int32_t> merge_priority_;

    // Special token IDs
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = -1;

    // Settings
    bool add_bos_ = true;

    // Byte fallback: byte value -> token ID for tokens of type BYTE
    std::unordered_map<uint8_t, int32_t> byte_to_id_;

    // Cache for known multi-character tokens that map directly
    // (for efficient lookup during initial tokenization)
    std::unordered_set<int32_t> byte_token_ids_;
};

} // namespace kaguya

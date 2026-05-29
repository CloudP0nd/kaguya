// Kaguya — BPE tokenizer implementation
// Reads tokenizer metadata from GGUF files and implements Byte Pair Encoding

#include "kaguya/tokenizer.h"

#include <algorithm>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <locale>
#include <codecvt>

namespace kaguya {

// ============================================================================
// UTF-8 helpers
// ============================================================================

/// Decode one UTF-8 codepoint from a string position.
/// Returns the codepoint and advances the index.
static uint32_t utf8_decode(const std::string& s, size_t& i) {
    uint8_t c0 = static_cast<uint8_t>(s[i]);

    if (c0 < 0x80) {
        // 1-byte: 0xxxxxxx
        i += 1;
        return c0;
    } else if ((c0 & 0xE0) == 0xC0) {
        // 2-byte: 110xxxxx 10xxxxxx
        if (i + 1 >= s.size()) { i = s.size(); return 0xFFFD; }
        uint8_t c1 = static_cast<uint8_t>(s[i + 1]);
        if ((c1 & 0xC0) != 0x80) { i += 1; return 0xFFFD; }
        uint32_t cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        i += 2;
        return cp;
    } else if ((c0 & 0xF0) == 0xE0) {
        // 3-byte: 1110xxxx 10xxxxxx 10xxxxxx
        if (i + 2 >= s.size()) { i = s.size(); return 0xFFFD; }
        uint8_t c1 = static_cast<uint8_t>(s[i + 1]);
        uint8_t c2 = static_cast<uint8_t>(s[i + 2]);
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) { i += 1; return 0xFFFD; }
        uint32_t cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        i += 3;
        return cp;
    } else if ((c0 & 0xF8) == 0xF0) {
        // 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        if (i + 3 >= s.size()) { i = s.size(); return 0xFFFD; }
        uint8_t c1 = static_cast<uint8_t>(s[i + 1]);
        uint8_t c2 = static_cast<uint8_t>(s[i + 2]);
        uint8_t c3 = static_cast<uint8_t>(s[i + 3]);
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) {
            i += 1; return 0xFFFD;
        }
        uint32_t cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12)
                    | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        i += 4;
        return cp;
    }

    // Invalid UTF-8 lead byte
    i += 1;
    return 0xFFFD;
}

/// Encode a Unicode codepoint to UTF-8
static std::string utf8_encode(uint32_t cp) {
    std::string result;
    if (cp < 0x80) {
        result += static_cast<char>(cp);
    } else if (cp < 0x800) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

/// Encode a single byte as a hex representation like <0xHH>
static std::string byte_to_hex_token(uint8_t b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    return buf;
}

// ============================================================================
// BpeTokenizer::build
// ============================================================================

bool BpeTokenizer::build(const std::vector<std::string>& tokens,
                          const std::vector<float>& scores,
                          const std::vector<int32_t>& token_types,
                          const std::vector<std::string>& merges,
                          int32_t bos_id,
                          int32_t eos_id,
                          bool add_bos)
{
    // Validate input sizes
    if (tokens.empty()) return false;

    const int32_t n = static_cast<int32_t>(tokens.size());

    // Check that scores and token_types have matching sizes
    // (they may be empty if the GGUF didn't contain them)
    bool has_scores = !scores.empty() && static_cast<int32_t>(scores.size()) == n;
    bool has_types = !token_types.empty() && static_cast<int32_t>(token_types.size()) == n;

    // Build id->token and token->id mappings
    id_to_token_.resize(n);
    id_to_score_.resize(n, 0.0f);
    id_to_type_.resize(n, TokenType::NORMAL);
    token_to_id_.clear();
    byte_to_id_.clear();
    byte_token_ids_.clear();

    for (int32_t i = 0; i < n; ++i) {
        id_to_token_[i] = tokens[i];

        if (has_scores) {
            id_to_score_[i] = scores[i];
        }

        if (has_types) {
            id_to_type_[i] = static_cast<TokenType>(token_types[i]);
        }

        token_to_id_[tokens[i]] = i;

        // Track byte tokens (type == BYTE)
        if (has_types && static_cast<TokenType>(token_types[i]) == TokenType::BYTE) {
            // Byte tokens have the form <0xHH>
            const auto& tok = tokens[i];
            if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x'
                && tok[5] == '>') {
                // Parse hex byte
                auto hex_val = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    return -1;
                };
                int hi = hex_val(tok[3]);
                int lo = hex_val(tok[4]);
                if (hi >= 0 && lo >= 0) {
                    uint8_t byte_val = static_cast<uint8_t>(hi * 16 + lo);
                    byte_to_id_[byte_val] = i;
                    byte_token_ids_.insert(i);
                }
            }
        }
    }

    // Build merge priority map
    merge_priority_.clear();
    for (int32_t i = 0; i < static_cast<int32_t>(merges.size()); ++i) {
        merge_priority_[merges[i]] = i; // Lower index = higher priority
    }

    // Detect BOS/EOS/PAD/UNK IDs
    // First, use the explicitly provided IDs
    bos_id_ = bos_id;
    eos_id_ = eos_id;

    // Then try to auto-detect from token names if not provided
    if (bos_id_ < 0) {
        // Try common BOS token names
        for (const char* name : {"<s>", "<|begin_of_text|>", "<|startoftext|>",
                                  "<BOS>", "<bos>", "[BOS]"}) {
            auto it = token_to_id_.find(name);
            if (it != token_to_id_.end()) {
                bos_id_ = it->second;
                break;
            }
        }
    }

    if (eos_id_ < 0) {
        // Try common EOS token names
        for (const char* name : {"</s>", "<|end_of_text|>", "<|endoftext|>",
                                  "<EOS>", "<eos>", "[EOS]", "<|end|>",
                                  "<|eot_id|>"}) {
            auto it = token_to_id_.find(name);
            if (it != token_to_id_.end()) {
                eos_id_ = it->second;
                break;
            }
        }
    }

    // Detect PAD and UNK
    pad_id_ = -1;
    for (const char* name : {"<pad>", "<|pad|>", "<PAD>", "[PAD]"}) {
        auto it = token_to_id_.find(name);
        if (it != token_to_id_.end()) {
            pad_id_ = it->second;
            break;
        }
    }

    unk_id_ = -1;
    // First check for UNKNOWN type tokens
    if (has_types) {
        for (int32_t i = 0; i < n; ++i) {
            if (id_to_type_[i] == TokenType::UNKNOWN) {
                unk_id_ = i;
                break;
            }
        }
    }
    if (unk_id_ < 0) {
        for (const char* name : {"<unk>", "<|unk|>", "<UNK>", "[UNK]"}) {
            auto it = token_to_id_.find(name);
            if (it != token_to_id_.end()) {
                unk_id_ = it->second;
                break;
            }
        }
    }

    add_bos_ = add_bos;

    return !id_to_token_.empty();
}

// ============================================================================
// BpeTokenizer::encode
// ============================================================================

std::vector<int32_t> BpeTokenizer::encode(const std::string& text,
                                            std::optional<bool> add_bos_opt,
                                            bool add_eos) const
{
    if (text.empty()) {
        std::vector<int32_t> result;
        bool should_add_bos = add_bos_opt.value_or(add_bos_);
        if (should_add_bos && bos_id_ >= 0) {
            result.push_back(bos_id_);
        }
        if (add_eos && eos_id_ >= 0) {
            result.push_back(eos_id_);
        }
        return result;
    }

    // Preprocess text (SentencePiece-style space handling)
    std::string processed = preprocess_text(text);

    // Convert text to initial token candidates
    std::vector<int32_t> initial_tokens = text_to_initial_tokens(processed);

    // Apply BPE merges
    std::vector<int32_t> merged = apply_bpe(initial_tokens);

    // Build final result
    std::vector<int32_t> result;
    result.reserve(merged.size() + 2); // extra for BOS/EOS

    bool should_add_bos = add_bos_opt.value_or(add_bos_);
    if (should_add_bos && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    result.insert(result.end(), merged.begin(), merged.end());

    if (add_eos && eos_id_ >= 0) {
        result.push_back(eos_id_);
    }

    return result;
}

// ============================================================================
// BpeTokenizer::decode
// ============================================================================

std::string BpeTokenizer::decode(const std::vector<int32_t>& tokens,
                                   bool skip_special) const
{
    std::string result;

    for (int32_t id : tokens) {
        result += decode_token(id, skip_special);
    }

    // Post-process: convert ▁ back to spaces (SentencePiece convention)
    // The ▁ character (U+2581, LOWER ONE EIGHTH BLOCK) is used by
    // SentencePiece to represent spaces between words.
    std::string final_result;
    final_result.reserve(result.size());

    for (size_t i = 0; i < result.size(); ) {
        // Check for ▁ (UTF-8: E2 96 81)
        if (i + 2 < result.size() &&
            static_cast<uint8_t>(result[i]) == 0xE2 &&
            static_cast<uint8_t>(result[i + 1]) == 0x96 &&
            static_cast<uint8_t>(result[i + 2]) == 0x81) {
            // Replace ▁ with space, but strip leading space at start
            if (!final_result.empty() || i > 0) {
                final_result += ' ';
            }
            i += 3;
        } else {
            final_result += result[i];
            i += 1;
        }
    }

    return final_result;
}

std::string BpeTokenizer::decode_token(int32_t token_id, bool skip_special) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(id_to_token_.size())) {
        return "";
    }

    // Skip special tokens if requested
    if (skip_special && is_special_token(token_id)) {
        return "";
    }

    const auto& text = id_to_token_[token_id];

    // For BYTE-type tokens, decode the byte value
    if (id_to_type_[token_id] == TokenType::BYTE) {
        // Byte tokens are stored as <0xHH>
        if (text.size() == 6 && text[0] == '<' && text[1] == '0' && text[2] == 'x'
            && text[5] == '>') {
            auto hex_val = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return -1;
            };
            int hi = hex_val(text[3]);
            int lo = hex_val(text[4]);
            if (hi >= 0 && lo >= 0) {
                return std::string(1, static_cast<char>(hi * 16 + lo));
            }
        }
    }

    // For CONTROL-type tokens, skip them
    if (id_to_type_[token_id] == TokenType::CONTROL && skip_special) {
        return "";
    }

    return text;
}

// ============================================================================
// BpeTokenizer::apply_bpe
// ============================================================================

std::vector<int32_t> BpeTokenizer::apply_bpe(const std::vector<int32_t>& initial_tokens) const {
    if (initial_tokens.size() <= 1) return initial_tokens;
    if (merge_priority_.empty()) return initial_tokens; // No merges available

    // Working list of tokens
    std::vector<int32_t> tokens = initial_tokens;

    // Iteratively merge the highest-priority pair
    while (tokens.size() > 1) {
        // Find the pair with the lowest merge priority (highest precedence)
        int32_t best_priority = INT32_MAX;
        size_t best_idx = 0;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            // Skip special tokens (they shouldn't be merged)
            if (is_special_token(tokens[i]) || is_special_token(tokens[i + 1])) {
                continue;
            }

            // Build the merge key: "token_a token_b"
            const auto& text_a = id_to_token_[tokens[i]];
            const auto& text_b = id_to_token_[tokens[i + 1]];
            std::string key = text_a + " " + text_b;

            auto it = merge_priority_.find(key);
            if (it != merge_priority_.end() && it->second < best_priority) {
                best_priority = it->second;
                best_idx = i;
            }
        }

        // No more merges possible
        if (best_priority == INT32_MAX) break;

        // Find the merged token
        const auto& text_a = id_to_token_[tokens[best_idx]];
        const auto& text_b = id_to_token_[tokens[best_idx + 1]];
        std::string merged_text = text_a + text_b;

        auto merged_it = token_to_id_.find(merged_text);
        if (merged_it == token_to_id_.end()) {
            // This shouldn't happen if the merge rules are consistent,
            // but handle it gracefully
            break;
        }

        // Replace the pair with the merged token
        tokens[best_idx] = merged_it->second;
        tokens.erase(tokens.begin() + static_cast<ptrdiff_t>(best_idx) + 1);
    }

    return tokens;
}

// ============================================================================
// BpeTokenizer::char_to_byte_tokens
// ============================================================================

std::vector<int32_t> BpeTokenizer::char_to_byte_tokens(uint32_t codepoint) const {
    std::vector<int32_t> result;

    // Encode the codepoint to UTF-8 bytes
    std::string utf8 = utf8_encode(codepoint);

    for (unsigned char byte : utf8) {
        // Look up the byte token
        auto it = byte_to_id_.find(byte);
        if (it != byte_to_id_.end()) {
            result.push_back(it->second);
        } else if (unk_id_ >= 0) {
            result.push_back(unk_id_);
        }
        // If no byte token and no UNK, skip the byte (shouldn't happen with proper GGUF)
    }

    return result;
}

// ============================================================================
// BpeTokenizer::preprocess_text
// ============================================================================

std::string BpeTokenizer::preprocess_text(const std::string& text) const {
    // SentencePiece-style preprocessing:
    // Replace spaces with ▁ (U+2581, LOWER ONE EIGHTH BLOCK)
    // This is how LLaMA and most modern models handle tokenization.
    //
    // The ▁ character represents word boundaries in SentencePiece.
    // "Hello world" → "▁Hello▁world"
    //
    // Note: We add ▁ at the start only if the first character isn't a space.
    // Multiple spaces are preserved as multiple ▁ characters.

    std::string result;
    result.reserve(text.size() + 8); // extra for ▁ characters

    // Prepend ▁ (SentencePiece convention: text starts with word boundary)
    const char sp[] = "\xE2\x96\x81"; // ▁ in UTF-8
    result.append(sp, 3);

    for (size_t i = 0; i < text.size(); ) {
        if (text[i] == ' ') {
            result.append(sp, 3);
            i += 1;
        } else {
            result += text[i];
            i += 1;
        }
    }

    return result;
}

// ============================================================================
// BpeTokenizer::text_to_initial_tokens
// ============================================================================

std::vector<int32_t> BpeTokenizer::text_to_initial_tokens(const std::string& text) const {
    std::vector<int32_t> result;

    // Try to match the longest known token at each position.
    // If no token matches, fall back to byte tokens for that character.
    size_t i = 0;
    while (i < text.size()) {
        bool found = false;

        // Try to match the longest possible token starting at position i
        // Maximum token length is typically around 64 characters
        size_t max_len = std::min(static_cast<size_t>(64), text.size() - i);

        for (size_t len = max_len; len >= 1; --len) {
            std::string candidate = text.substr(i, len);
            auto it = token_to_id_.find(candidate);

            if (it != token_to_id_.end()) {
                // Found a direct match
                result.push_back(it->second);
                i += len;
                found = true;
                break;
            }
        }

        if (!found) {
            // No token matches this position.
            // Decode the current UTF-8 character and use byte fallback.
            size_t start_i = i;
            uint32_t cp = utf8_decode(text, i);

            if (cp == 0xFFFD && i == start_i) {
                // Truly invalid byte, skip it
                i += 1;
                continue;
            }

            auto byte_tokens = char_to_byte_tokens(cp);
            result.insert(result.end(), byte_tokens.begin(), byte_tokens.end());
        }
    }

    return result;
}

// ============================================================================
// BpeTokenizer::accessors
// ============================================================================

const std::string& BpeTokenizer::token_text(int32_t id) const {
    static const std::string empty = "";
    if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) {
        return empty;
    }
    return id_to_token_[id];
}

bool BpeTokenizer::is_special_token(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(id_to_type_.size())) {
        return false;
    }
    auto type = id_to_type_[id];
    return type == TokenType::CONTROL || type == TokenType::UNKNOWN;
}

} // namespace kaguya

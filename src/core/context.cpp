#include "kaguya/context.h"

namespace kaguya {

Context::Context(const Model& model)
    : model_(model), hparams_(model.hparams()), pos_(0) {}

void Context::reset() {
    pos_ = 0;
}

} // namespace kaguya

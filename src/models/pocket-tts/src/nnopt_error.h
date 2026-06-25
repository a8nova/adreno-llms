#pragma once
// nnopt_error.h — error/diagnostic macros for op implementations.
//
// The scaffold's per-op files (src/ops/*.cpp) include "../nnopt_error.h", but
// the NNOPT_ERROR / NNOPT_ERROR_FMT / NNOPT_CHECKPOINT / NNOPT_LAYER_CHECK
// macros actually live in debug_utils.h (kept together with the crash handler
// and layer-dump machinery). This thin shim forwards there so the ops compile
// without duplicating the macro definitions.
#include "debug_utils.h"

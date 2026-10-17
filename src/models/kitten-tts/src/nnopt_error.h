#pragma once
// Compatibility shim: scaffold-emitted op stubs include "../nnopt_error.h"
// but the NNOPT_ERROR / NNOPT_ERROR_FMT / NNOPT_LAYER_* macros actually live
// in debug_utils.h. This header re-exports them so every src/ops/*.cpp that
// includes "../nnopt_error.h" compiles unchanged.
#include "debug_utils.h"

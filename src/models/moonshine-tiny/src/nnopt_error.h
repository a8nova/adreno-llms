#pragma once
// nnopt_error.h — thin alias header. The Moonshine ops were authored to
// include "nnopt_error.h" for the NNOPT_ERROR / NNOPT_ERROR_FMT /
// NNOPT_CHECKPOINT macros, which actually live in debug_utils.h. Keep this
// header so those includes resolve without editing every op file.
#include "debug_utils.h"

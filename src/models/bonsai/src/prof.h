// GPU per-kernel profiler — uses cl_event profiling to attribute decode time
// to individual kernel names. Per the Snapdragon programming guide section
// 4.5.2: enable CL_QUEUE_PROFILING_ENABLE, attach an event to each
// clEnqueueNDRangeKernel, query CL_PROFILING_COMMAND_(START|END) for kernel
// duration. Aggregate by kernel name → bottleneck table.
//
// Active only when NNOPT_PROFILE=1 (env). When disabled, the wrapper is a
// straight pass-through with zero overhead.
//
// Pattern: deferred-readout — every dispatch saves its event into a per-name
// list. nnopt_prof::dump() syncs the queue once, then walks the lists. Saves
// the per-launch synchronous wait that would otherwise destroy the async
// queue's pipelining.
#pragma once

#include <CL/cl.h>

namespace nnopt_prof {

// One-time init from NNOPT_PROFILE env var. Cheap, idempotent.
bool enabled();

// Drop-in replacement for clEnqueueNDRangeKernel. When profiling is on,
// records a (kernel_name, event) pair internally. Caller may still pass a
// non-null cl_event*; the recorded event is independent.
cl_int enqueue(cl_command_queue queue,
               cl_kernel        kernel,
               cl_uint          work_dim,
               const size_t*    global_work_offset,
               const size_t*    global_work_size,
               const size_t*    local_work_size,
               cl_uint          num_events_in_wait_list,
               const cl_event*  event_wait_list,
               cl_event*        event_out);

// Sync the live queue, query every recorded event, print sorted-by-total
// time table to stderr. Clears the recording state. Call from generate()
// after the decode loop completes.
void dump(cl_command_queue queue);

// Reset (drop all recordings without dumping). Used between prefill and
// decode if you want to isolate decode-only attribution.
void reset();

} // namespace nnopt_prof

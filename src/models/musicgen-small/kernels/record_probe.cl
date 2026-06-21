// record_probe.cl — cl_qcom_recordable_queues validation (NNOPT_RECORD_PROBE=1).
// Recipe from a8nova/adreno-llms (validated on this Razr 2020 / Adreno 620):
// queue created with CL_QUEUE_RECORDABLE_QCOM (0x40000000) ALONE.
__kernel void probe_noop(__global int* x) {
  if (get_global_id(0) == 0) x[0] = x[0] + 1;
}

// Arg-override semantics probe: writes `v` at slot get_group_id — replaying
// with a scalar override on arg 1 shows whether the override (a) is accepted,
// (b) applies to ALL instances of the kernel inside one recording.
__kernel void probe_scalar(__global int* x, const int v) {
  if (get_local_id(0) == 0) x[get_group_id(0) + 1] = v;
}

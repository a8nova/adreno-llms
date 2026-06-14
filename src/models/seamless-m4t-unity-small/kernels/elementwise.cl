// Element-wise activation + residual helpers. (preamble provides storage_t/act_apply)
__kernel void act_inplace(__global storage_t* buf, const int n, const int code) {
    int gid = get_global_id(0);
    if (gid < n) STORE(buf, gid, act_apply((float)LOAD(buf, gid), code));
}
// a[i] += alpha * b[i]
__kernel void axpy_inplace(__global storage_t* a, __global const storage_t* b,
                           const float alpha, const int n) {
    int gid = get_global_id(0);
    if (gid < n) STORE(a, gid, (float)LOAD(a, gid) + alpha * (float)LOAD(b, gid));
}
// a[i] *= alpha
__kernel void scale_inplace(__global storage_t* a, const float alpha, const int n) {
    int gid = get_global_id(0);
    if (gid < n) STORE(a, gid, alpha * (float)LOAD(a, gid));
}

// dot8_probe.cl — one-shot semantics probe for qcom_dot8_acc(uint, uint, int).
// Discriminating vectors: each output uniquely identifies an interpretation.
// Run via NNOPT_DOT8_PROBE=1 (host prints the table and exits).
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable

__kernel void dot8_probe(__global int* out) {
  if (get_global_id(0) != 0) return;
  // [0] bytes a=(1,2,3,4) · b=(1,1,1,1):
  //     byte-signed/unsigned → 10; nibble path → a=(1,0,2,0,3,0,4,0)·(1,0,1,0,1,0,1,0)=10 too
  out[0] = qcom_dot8_acc(0x04030201u, 0x01010101u, 0);
  // [1] a byte0=0xFF · b byte0=1: byte-signed → -1; byte-unsigned → 255;
  //     nibble-signed → (15? -1?, -1?) nibbles (0xF,0xF)·(1,0) = f(-1)*1 = -1 or 15
  out[1] = qcom_dot8_acc(0x000000FFu, 0x00000001u, 0);
  // [2] all-0xFF · all-ones-bytes: byte-signed → -4; byte-unsigned → 1020
  out[2] = qcom_dot8_acc(0xFFFFFFFFu, 0x01010101u, 0);
  // [3] accumulator check: 1·1 + acc(100) → 101
  out[3] = qcom_dot8_acc(0x00000001u, 0x00000001u, 100);
  // [4] 0xFF·0xFF single byte: both-signed → 1; both-unsigned → 65025; mixed → -255
  out[4] = qcom_dot8_acc(0x000000FFu, 0x000000FFu, 0);
  // [5] NIBBLE discriminator: a=0x21, b=0x11.
  //     byte: 0x21(33)·0x11(17) = 561. nibble: (1,2)·(1,1) = 3.
  out[5] = qcom_dot8_acc(0x00000021u, 0x00000011u, 0);
  // [6] nibble order/sign: a=0xF1, b=0x11. byte: 241*17=4097 (s: -15*17=-255).
  //     nibble-signed: (1,-1)·(1,1) = 0; nibble-unsigned: (1,15)·(1,1) = 16.
  out[6] = qcom_dot8_acc(0x000000F1u, 0x00000011u, 0);
  // [7] cross-uint check via second arg high byte: a=(0,0,0,2)·b=(0,0,0,3) = 6
  //     (confirms byte lanes align, not shifted)
  out[7] = qcom_dot8_acc(0x02000000u, 0x03000000u, 0);
}

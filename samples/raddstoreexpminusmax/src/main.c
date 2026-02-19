// Test harness for f32-raddstoreexpminusmax (float32 reduce-add-store-exp-minus-max)
// RVV implementation test

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <riscv_vector.h>
// #include <xnnpack.h> // Include XNNPack headers
// #include <zephyr/arch/cpu.h>
// #include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>


#define BATCH_SIZE 256
#define OUTPUT_MATRIX_NAME output_arr

static float input_arr[BATCH_SIZE];
static float output_arr[BATCH_SIZE];
static float gold[BATCH_SIZE];
static float max_val;
static float sum_val;
static float gold_sum;

#define REPEAT_TEST_ITERS 1

void reference_f32_raddstoreexpminusmax(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params)
{

  const float xmin = -0x1.5ebb82p6;
  const float r_ln2f = 0x1.715476p+0f;
  const float l2uf = 0x1.62E400p-1f;
  const float l2lf = 0x1.7F7D1Cp-20f;
  const float c6 = 0x1.6850e4p-10f;
  const float c5 = 0x1.123bccp-7;
  const float c4 = 0x1.555b98p-5f;
  const float c3 = 0x1.55548ep-3f;
  const float c2 = 0x1.fffff8p-2f;
  const int16_t p = (24 - 1);
  const int16_t bias = (128 - 1);

  size_t n = batch >> 2;
  size_t avl = n;
  size_t vl = __riscv_vsetvl_e32m4(n);

  vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  do {
    vl = __riscv_vsetvl_e32m4(avl);
    avl -= vl;
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, *max, vl);
    input += vl;

    vx = __riscv_vfmax_vf_f32m4(vx, xmin, vl);

    vfloat32m4_t v = __riscv_vfmul_vf_f32m4(vx, r_ln2f, vl);
    vint16m2_t q = __riscv_vfncvt_x_f_w_i16m2(v, vl);
    vfloat32m4_t z = __riscv_vfwcvt_f_x_v_f32m4(q, vl);

    vfloat32m4_t s = __riscv_vfnmsac_vf_f32m4(vx, l2uf, z, vl);
    s = __riscv_vfnmsac_vf_f32m4(s, l2lf, z, vl);

    vfloat32m4_t poly_z;
    vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(c5, vl);
    y = __riscv_vfmacc_vf_f32m4(y, c6, s, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c4, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c3, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c2, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    vint32m4_t qw = __riscv_vwadd_vx_i32m4(q, bias, vl);
    vint32m4_t qq = __riscv_vsll_vx_i32m4(qw, p, vl);
    vfloat32m4_t qf = __riscv_vreinterpret_v_i32m4_f32m4(qq);
    vfloat32m4_t vexp = __riscv_vfmul_vv_f32m4(y, qf, vl);

    __riscv_vse32_v_f32m4(output, vexp, vl);
    output += vl;
    vsum = __riscv_vfadd_vv_f32m4_tu(vsum, vsum, vexp, vl);
    printf("avl: %zu\n", avl);
  } while(avl > 0);

  vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
//   printf("vl: %zu\n", vl);
  printf("vsum\n");
  vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(vsum, v0, n);
  printf("sum = %p\n", sum);
  *sum = __riscv_vfmv_f_s_f32m1_f32(red);
  printf("sum\n");
}


int full_is_equal(float* x, float* y) {
    size_t n = BATCH_SIZE;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(x[i] - y[i]);
        float mag = fmaxf(fabsf(x[i]), fabsf(y[i]));
        // Use relative tolerance for exp values which can span wide range
        if (mag > 1e-6f) {
            if (diff / mag > 1e-4f) {
                printf("Mismatch at index %zu: got %f, expected %f\n", i, x[i], y[i]);
                return 0;
            }
        } else {
            if (diff > 1e-6f) {
                printf("Mismatch at index %zu: got %f, expected %f\n", i, x[i], y[i]);
                return 0;
            }
        }
    }
    return 1;
}

unsigned long read_cycles() {
    unsigned long cc;
    __asm__ volatile("rdcycle %0" : "=r"(cc));
    return cc;
}

#define gemmini_flush(x) do {} while(0) 


#define RESET_STATE() do { \
    batch = BATCH_SIZE * sizeof(float); \
    input = input_arr; \
    output = output_arr; \
    sum_val = 0.0f; \
    sum = &sum_val; \
} while(0)

static inline void fence(void) {
    __asm__ volatile("fence" ::: "memory");
}

static float rand_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}


__attribute__((noinline)) void run_candidate_0(size_t batch, const float* input, const float* max, float* output, float* sum, const void* params) {


  const float xmin = -0x1.5ebb82p6;
  const float r_ln2f = 0x1.715476p+0f;
  const float l2uf = 0x1.62E400p-1f;
  const float l2lf = 0x1.7F7D1Cp-20f;
  const float c6 = 0x1.6850e4p-10f;
  const float c5 = 0x1.123bccp-7;
  const float c4 = 0x1.555b98p-5f;
  const float c3 = 0x1.55548ep-3f;
  const float c2 = 0x1.fffff8p-2f;
  const int16_t p = (24 - 1);
  const int16_t bias = (128 - 1);

  size_t n = batch >> 2;
  size_t avl = n;
  size_t vl = __riscv_vsetvl_e32m4(n);

  vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  do {
    vl = __riscv_vsetvl_e32m4(avl);
    avl -= vl;
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, *max, vl);
    input += vl;

    vx = __riscv_vfmax_vf_f32m4(vx, xmin, vl);

    vfloat32m4_t v = __riscv_vfmul_vf_f32m4(vx, r_ln2f, vl);
    vint16m2_t q = __riscv_vfncvt_x_f_w_i16m2(v, vl);
    vfloat32m4_t z = __riscv_vfwcvt_f_x_v_f32m4(q, vl);

    vfloat32m4_t s = __riscv_vfnmsac_vf_f32m4(vx, l2uf, z, vl);
    s = __riscv_vfnmsac_vf_f32m4(s, l2lf, z, vl);

    vfloat32m4_t poly_z;
    vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(c5, vl);
    y = __riscv_vfmacc_vf_f32m4(y, c6, s, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c4, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c3, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(c2, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    poly_z = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, s, poly_z, vl);

    vint32m4_t qw = __riscv_vwadd_vx_i32m4(q, bias, vl);
    vint32m4_t qq = __riscv_vsll_vx_i32m4(qw, p, vl);
    vfloat32m4_t qf = __riscv_vreinterpret_v_i32m4_f32m4(qq);
    vfloat32m4_t vexp = __riscv_vfmul_vv_f32m4(y, qf, vl);

    __riscv_vse32_v_f32m4(output, vexp, vl);
    output += vl;
    vsum = __riscv_vfadd_vv_f32m4_tu(vsum, vsum, vexp, vl);
  } while(avl > 0);

  vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    *sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsum, v0, n));


}

int main() {
    for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
        // Initialize input data
        max_val = -INFINITY;
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            input_arr[i] = rand_float();
            if (input_arr[i] > max_val) {
                max_val = input_arr[i];
            }
        }

        // Clear output
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            output_arr[i] = 0.0f;
        }

        // Compute gold reference
        gold_sum = 0.0f;
        reference_f32_raddstoreexpminusmax(BATCH_SIZE * sizeof(float), input_arr, &max_val, gold, &gold_sum, NULL);

        // Clear output again for candidate
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            output_arr[i] = 0.0f;
        }

        // Set up variables for injected code
        size_t batch = BATCH_SIZE * sizeof(float);
        const float* input = input_arr;
        const float* max = &max_val;
        float* output = output_arr;
        sum_val = 0.0f;
        float* sum = &sum_val;
        const void* params = NULL;

        // SUBSTITUTE HERE

    unsigned long start_cycle, end_cycle;


    // Run candidate 0
    RESET_STATE();
    fence();
    __asm__ volatile("vsetvli x0, x0, e8, m1, ta, ma");
    start_cycle = read_cycles();
    run_candidate_0(batch, input, max, output, sum, params);
    fence();
    __asm__ volatile("vsetvli x0, x0, e8, m1, ta, ma");
    end_cycle = read_cycles();
    printf("ID 0 latency: %lu cycles\n", end_cycle - start_cycle);


        // SUBSTITUTE END

        // Verify results
        if (!full_is_equal(output_arr, gold)) {
            printf("FAIL: Output mismatch\n");
            sys_reboot(SYS_REBOOT_COLD);
        }
        // Verify sum
        float sum_diff = fabsf(sum_val - gold_sum);
        float sum_mag = fmaxf(fabsf(sum_val), fabsf(gold_sum));
        if (sum_mag > 1e-6f && sum_diff / sum_mag > 1e-4f) {
            printf("FAIL: Sum mismatch: got %f, expected %f\n", sum_val, gold_sum);
            sys_reboot(SYS_REBOOT_COLD);
        }
    }
    printf("Correct result\n");
    sys_reboot(SYS_REBOOT_COLD);
}
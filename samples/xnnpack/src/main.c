/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <pthreadpool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/_intsup.h>
#include <xnnpack.h> // Include XNNPack headers
#include <zephyr/arch/cpu.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

const size_t batch_size = 16; // the test is only for batch size 1
const size_t input_channels = 2048;
const size_t output_channels = 256;

unsigned long cycle()
{
	unsigned long cc;
	__asm__ volatile("rdcycle  %0" : "=r"(cc));
	return cc;
}

volatile int wait = 0;
int main(void)
{
	printf("Hello World! %s\n", CONFIG_BOARD_TARGET);
	printf("Zephyr is running on %d CPUs\n", CONFIG_MP_MAX_NUM_CPUS);
	printf("Running XNNPACK FP32 Test\n");
	// while(wait == 0); // used for debugging

	// Test malloc
	void *test_alloc = aligned_alloc(0x40, 0x380);
	test_alloc = aligned_alloc(0x40, 0x380);
	if (!test_alloc) {
		printf("Test malloc failed!\n");
		return -1;
	} else {
		printf("Test malloc succeeded!\n");
		free(test_alloc);
	}

	// create pthreadpool
	pthreadpool_t threadpool = NULL;
	threadpool = pthreadpool_create(0);

	if (threadpool == NULL) {
		printf("Failed to create pthreadpool\n");
		return -1;
	} else {
		printf("pthreadpool created successfully!\n");
	}

	// Initialize XNNPACK
	int status = xnn_initialize(NULL);
	if (status != xnn_status_success) {
		printf("Failed to initialize XNNPack, status code: %d\n", status);
		return -1;
	}
	printf("XNNPACK initialized successfully!\n");

	int8_t *input_data = (int8_t *)malloc(input_channels * batch_size * sizeof(int8_t));
	int8_t *weights = (int8_t *)malloc(input_channels * output_channels * sizeof(int8_t));
	float *scale = (float *)malloc(output_channels * sizeof(float));
	int32_t *bias = (int32_t *)malloc(output_channels * sizeof(int32_t));
	int8_t *output_data = (int8_t *)malloc(output_channels * batch_size * sizeof(int8_t));
	int8_t *output_data_ref = (int8_t *)malloc(output_channels * batch_size * sizeof(int8_t));
	
	int8_t minzp = -128;
	int8_t maxzp = 127;
	int8_t outzp = 0;

	printf("Test shapes: %zu, %zu\n", input_channels, output_channels);

	printf("Preparing input data and weights\n");
	// Initialize input data
	int8_t zero_point = 0;
	for (size_t i = 0; i < input_channels * batch_size; i++) {
		input_data[i] = (int8_t)(i%128);
	}
	// Initialize weights
	for (size_t i = 0; i < input_channels * output_channels; i++) {
		weights[i] = (int8_t)((i * i)%128);
	}
	for (size_t i = 0; i < output_channels; i++) {
		scale[i] = (float)1.0f;
		bias[i] = (int32_t)i;
	}
	// Compute reference output
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t i = 0; i < output_channels; i++) {
			output_data[i + output_channels*b] = 0;
			int32_t acc = (int32_t) bias[i];
			for (size_t j = 0; j < input_channels * batch_size; j++) {
				acc += ((int32_t)input_data[j]) * (int32_t)weights[i * input_channels + j];
			}
			float facc = scale[i] * (float)acc;
			output_data_ref[i + output_channels*b] = (int8_t)fmaxf(fminf(facc, 127.0f), -128.0f);
		}
	}

	printf("Creating operators\n");
	// Create the Fully Connected operator
	xnn_operator_t fc_op = NULL;
	status = xnn_create_fully_connected_nc_qs8_qc8w(
						   input_channels,  // Input size per batch
						   output_channels, // Output size per batch
						   input_channels,  // Input stride
						   output_channels, // Output stride
						   0,			   	// Input zero point
						   1.0f,			// Input scale
						   scale,	 		// kernel scale vector
						   weights,         // Weights matrix
						   bias,            // Bias vector
						   outzp,			   	// output zero point
  						   1.0f,         	// Output scale
						   minzp,       // Min activation
						   maxzp,        // Max activation
						   0,               // Flags
						   NULL,            // Code cache
						   NULL,            // Weights cache
						   &fc_op);

	if (status != xnn_status_success) {
		printf("Failed to create Fully Connected operator, status code: %d\n", status);
		return -1;
	}

	// Reshape the operator
	status = xnn_reshape_fully_connected_nc_qs8_qc8w(fc_op, batch_size, threadpool);
	if (status != xnn_status_success) {
		printf("Failed to reshape Fully Connected operator, status code: %d\n", status);
		xnn_delete_operator(fc_op);
		return -1;
	}

	// Setup the operator
	status = xnn_setup_fully_connected_nc_qs8_qc8w(fc_op, input_data, output_data);
	if (status != xnn_status_success) {
		printf("Failed to setup Fully Connected operator, status code: %d\n", status);
		xnn_delete_operator(fc_op);
		return -1;
	}

	printf("Shape of input and output data: %d, %d\n", input_channels, output_channels);

	unsigned long clock_start = cycle();

	// Run the operator
	status = xnn_run_operator(fc_op, threadpool);
	if (status != xnn_status_success) {
		printf("Failed to run Fully Connected operator, status code: %d\n", status);
		xnn_delete_operator(fc_op);
		return -1;
	}

	unsigned long clock_end = cycle();
	printf("Clocks taken: %ld\n", (clock_end - clock_start));

	/* // Print results
	printf("Fully Connected Output:\n");
	for (size_t i = 0; i < output_channels; i++)
	{
		printf("%f ", (double)output_data[i]); // Explicitly cast to avoid warnings
	}
	printf("\n"); */

	// Verify the output
	for (size_t i = 0; i < output_channels * batch_size; i++) {
		float diff = output_data[i] - output_data_ref[i];
		if (diff != 0) {
			printf("Output verification failed at index %zu: expected %d, got %d\n", i,
			       output_data_ref[i], output_data[i]);
		}
	}

	// Cleanup
	xnn_delete_operator(fc_op);

	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
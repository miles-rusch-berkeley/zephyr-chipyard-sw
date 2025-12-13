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
#include "xnnpack/operator.h"
#include <zephyr/arch/cpu.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

const size_t batch_size = 16;
const size_t input_channels = 4;
const size_t output_channels = 32;

unsigned long cycle()
{
	unsigned long cc;
	__asm__ volatile("rdcycle  %0" : "=r"(cc));
	return cc;
}

int main(void)
{
	printf("Target: %s\n", CONFIG_BOARD_TARGET);
	printf("CPUs: %d\n", CONFIG_MP_MAX_NUM_CPUS);
	printf("XNNPACK QS8\n");

	// create pthreadpool
	pthreadpool_t threadpool = NULL;
	threadpool = pthreadpool_create(0);

	if (threadpool == NULL) {
		printf("Failed to create pthreadpool\n");
		return -1;
	}

	// Initialize XNNPACK
	int status = xnn_initialize(NULL);
	if (status != xnn_status_success) {
		printf("Failed to initialize XNNPack, status code: %d\n", status);
		return -1;
	}

	int8_t *input_data = (int8_t *)malloc(batch_size * input_channels * sizeof(int8_t));
	int8_t *weights = (int8_t *)malloc(input_channels * output_channels * sizeof(int8_t));
	float *scale = (float *)malloc(output_channels * sizeof(float));
	int32_t *bias = (int32_t *)malloc(output_channels * sizeof(int32_t));
	int8_t *output_data = (int8_t *)malloc(batch_size * output_channels * sizeof(int8_t));
	int8_t *output_data_ref = (int8_t *)malloc(batch_size * output_channels * sizeof(int8_t));
	
	int8_t minzp = -128;
	int8_t maxzp = 127;
	int8_t outzp = 0;

	printf("chIn: %zu, chOut: %zu, bSz: %zu\n", input_channels, output_channels, batch_size);

	// Initialize input data
	int8_t zero_point = 0;
	for (size_t i = 0; i < batch_size * input_channels; i++) {
		// input_data[i] = (int8_t)(1);
		input_data[i] = (int8_t)(i - (batch_size*input_channels>>1));
	}
	// Initialize weights
	for (size_t i = 0; i < input_channels * output_channels; i++) {
		// weights[i] = (int8_t)((i - ((input_channels*output_channels)>>1)));
		weights[i] = (int8_t)(1);
	}
	for (size_t i = 0; i < output_channels; i++) {
		scale[i] = (float)1.0f;
		bias[i] = (int32_t)(i - (output_channels>>1));
	}
	// Compute reference output
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t i = 0; i < output_channels; i++) {
			output_data_ref[b * output_channels + i] = 0;
			int32_t acc = (int32_t) bias[i];
			for (size_t j = 0; j < input_channels; j++) {
				acc += ((int32_t)input_data[b * input_channels + j]) * (int32_t)weights[i * input_channels + j];
			}
			float facc = scale[i] * (float)acc;
			output_data_ref[b * output_channels + i] = (int8_t)fmaxf(fminf(facc, 127.0f), -128.0f);
		}
	}
	xnn_operator_t fc_opu = NULL;
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
		&fc_opu);
	if (status != xnn_status_success) {
		printf("Failed to create Fully Connected operator, status code: %d\n", status);
		return -1;
	}
	// Reshape the operator
	status = xnn_reshape_fully_connected_nc_qs8_qc8w(fc_opu, batch_size, threadpool);
	if (status != xnn_status_success) {
		printf("Failed to reshape Fully Connected operator, status code: %d\n", status);
		xnn_delete_operator(fc_opu);
		return -1;
	}
	// Setup the operator
	status = xnn_setup_fully_connected_nc_qs8_qc8w(fc_opu, input_data, output_data);
	if (status != xnn_status_success) {
		printf("Failed to setup Fully Connected operator, status code: %d\n", status);
		xnn_delete_operator(fc_opu);
		return -1;
	}

	// Run the operator
	unsigned long clock_start = cycle();
	status = xnn_run_operator(fc_opu, threadpool);
	if (status != xnn_status_success) {
		printf("Failed to run Fully Connected operator, status code: %d\n", status);
		xnn_delete_operator(fc_opu);
		return -1;
	}
	unsigned long clock_end = cycle();
	printf("Clocks taken (opu): %ld\n", (clock_end - clock_start));

	// Verify the output
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t i = 0; i < output_channels; i++) {
			int8_t diff = output_data[b * output_channels + i] - output_data_ref[b * output_channels + i];
			if (diff != 0) {
				printf("failed verification at index %zu, batch %zu: expected %d, got %d\n", i, b,
					output_data_ref[b * output_channels + i], output_data[b * output_channels + i]);
				
					printf("opu:\n");
					for (size_t bb = 0; bb < batch_size; bb++) {
						for (size_t ii = 0; ii < output_channels; ii++) {
							printf("%d ", output_data[bb * output_channels + ii]);
						}
						printf("\n");
					}
					printf("reference:\n");
					for (size_t bb = 0; bb < batch_size; bb++) {
						for (size_t ii = 0; ii < output_channels; ii++) {
							printf("%d ", output_data_ref[bb * output_channels + ii]);
						}
						printf("\n");
					}
					xnn_delete_operator(fc_opu);
					sys_reboot(SYS_REBOOT_COLD);
					return -1;
			}
		}
	}
	printf("passed verification!\n");

	xnn_delete_operator(fc_opu);
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
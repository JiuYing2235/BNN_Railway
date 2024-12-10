#ifndef LAYER_H
#define LAYER_H

#include "typedefs.h"
#include "dimension_def.h"
#include "pgconv.h"
#include "weights_fracnet_64.h"
#include "conv_weights.h"
#include <iostream>
#include <cstdlib>


//---------------------
//   util functions
//---------------------

/* 2-bit Quantization function  */
inline uint2 to2bit(FIX_FM_acc x)
{
	const FIX_WT scale = 1.5;
	ap_ufixed<2, 2, AP_RND, AP_SAT> temp = (ap_ufixed<2, 2, AP_RND, AP_SAT>)((x+1)*scale);
	return (uint2)temp;
}

/* 
Quantize feature maps to 2 bits and do bit-packing
on MSB and LSB, respectively
*/
void quant_and_pack(
		FIX_FM_acc prior_outputs[CHANNEL_OUT/CHANNEL_OUT_T][CHANNEL_OUT_T][WIDTH][WIDTH],
		uint64 msb_buffer[CHANNEL_IN][WIDTH][WIDTH],
		int H_fmap,
		int in_channels
)
{
	uint2 val = 0;
	LOOP_Quant_Pack:
	for (int ch_t=0; ch_t<CHANNEL_OUT/CHANNEL_OUT_T; ch_t++){
		for (int row=0; row<H_fmap; row++){
			for (int col=0; col<H_fmap; col++){
#pragma HLS PIPELINE
				int channel_start = ch_t*CHANNEL_OUT_T;
				for (int ch_offset=0; ch_offset<CHANNEL_OUT_T; ch_offset++){
					FIX_FM_acc raw_input = prior_outputs[ch_t][ch_offset][row][col];
					int ch = channel_start + ch_offset;
					if (ch < in_channels) {
						val = to2bit(raw_input);
					} else {
						val = 0;
					}
					msb_buffer[0][row][col][63 - ch] = val[1];
					msb_buffer[1][row][col][63 - ch] = val[0];
				}
			}
		}
	}
}

inline void pack_and_multiply(
    int16 block_t0[CHANNEL_OUT_T][WIDTH][WIDTH],
    const FIX_WT *bn_weight,
    int row,
    int col,
    int ch,
    ap_fixed<24, 16, AP_RND, AP_SAT> &result1,
    ap_fixed<24, 16, AP_RND, AP_SAT> &result2
) {
    ap_int<1> block_sign1 = (block_t0[ch][row + 1][col + 1] < 0) ? 1 : 0;
    ap_int<1> block_sign2 = (block_t0[ch + 1][row + 1][col + 1] < 0) ? 1 : 0;
    ap_int<1> weight_sign1 = (bn_weight[ch] < 0) ? 1 : 0;
    ap_int<1> weight_sign2 = (bn_weight[ch + 1] < 0) ? 1 : 0;

    FIX_WT weight_val1 = bn_weight[ch];
    if (weight_sign1) {
        weight_val1 = -weight_val1;
    }
    FIX_WT weight_val2 = bn_weight[ch + 1];
    if (weight_sign2) {
        weight_val2 = -weight_val2;
    }

    int36 block_pack = ((int36(abs(block_t0[ch][row + 1][col + 1])) & 0xFFF) << 24) | (int36(abs(block_t0[ch + 1][row + 1][col + 1])) & 0xFFF);

    int36 weight_pack = ((int36(weight_val1.range(11, 0)) & 0xFFF) << 24) | (int36(weight_val2.range(11, 0)) & 0xFFF);

    ap_int<72> result_pack = block_pack * weight_pack;

    result1 = result_pack.range(71, 48);
    result2 = result_pack.range(23, 0);

    result1 = result1 >> 8;
    result2 = result2 >> 8;

    if (block_sign1 != weight_sign1) {
        result1 = -result1;
    }
    if (block_sign2 != weight_sign2) {
        result2 = -result2;
    }
}


void bn1(
    FIX_FM_acc out_buf[CHANNEL_OUT / CHANNEL_OUT_T][CHANNEL_OUT_T][WIDTH][WIDTH],
    int16 block_t0[CHANNEL_OUT_T][WIDTH][WIDTH],
    const FIX_WT *bn_weight,
    const FIX_WT *bn_bias,
    int stride,
    int channel_tile,
    int H_fmap
) {
    for (int row = 0; row < H_fmap; row++) {
        for (int col = 0; col < H_fmap; col++) {
            for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch += 2) {
				ap_fixed<24, 16, AP_RND, AP_SAT> result1, result2;
				pack_and_multiply(block_t0, bn_weight, row, col, ch, result1, result2);

				out_buf[channel_tile][ch][row][col] = FIX_FM_acc(result1) + bn_bias[ch];
				out_buf[channel_tile][ch + 1][row][col] = FIX_FM_acc(result2) + bn_bias[ch + 1];
                }
            }
        }
    }

/* Batchnorm+activation+shortcut functions for conv layers other than the input layer */
void bn_relu_shortcut(
		FIX_FM_acc residual[CHANNEL_OUT/CHANNEL_OUT_T][CHANNEL_OUT_T][WIDTH][WIDTH],
		int16 block_t0[CHANNEL_OUT_T][WIDTH][WIDTH],
		int16 block_t1[CHANNEL_OUT_T][WIDTH][WIDTH],

		const FIX_WT *bn_weight_0,
		const FIX_WT *bn_weight_1,
		const FIX_WT *bn_bias_0,
		const FIX_WT *bn_bias_1,
		const FIX_WT *relu_weight,

		int stride,
		int channel_tile,
		int H_fmap,
		int out_channels
)
{


	FIX_FM_acc out_feature_t0[BN_CHANNEL_PARALLELISM];
	FIX_FM_acc out_feature_t1[BN_CHANNEL_PARALLELISM];
#pragma HLS ARRAY_PARTITION variable=out_feature_t0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_feature_t1 complete dim=1

	const FIX_WT msb_scale = 2.0/3.0;
	const FIX_WT lsb_scale = 1.0/3.0;

	LOOP_BN_RELU_SC:
	for (int i=0; i<H_fmap; i++) {
		for (int j=0; j<H_fmap; j++) {
#pragma HLS PIPELINE

			// Merge Tile
			for (int ch_offset=0; ch_offset<BN_CHANNEL_PARALLELISM; ch_offset++){
				out_feature_t0[ch_offset] = block_t0[ch_offset][i*stride+1][j*stride+1];
				out_feature_t1[ch_offset] = block_t1[ch_offset][i*stride+1][j*stride+1];
			}
			for (int ch_offset=0; ch_offset<BN_CHANNEL_PARALLELISM; ch_offset++){
				out_feature_t0[ch_offset] *= msb_scale;
				out_feature_t0[ch_offset] += (out_feature_t1[ch_offset]*lsb_scale);

			}

			// Load Residual
			for (int channel_pt=0; channel_pt<BN_CHANNEL_PARALLELISM; channel_pt++) {
				out_feature_t1[channel_pt] = residual[channel_tile][channel_pt][i][j];
			}

			// Batch Normalization
			for (int channel_pt=0; channel_pt<BN_CHANNEL_PARALLELISM; channel_pt++) {
				out_feature_t0[channel_pt] = bn_weight_0[channel_pt]*out_feature_t0[channel_pt] + bn_bias_0[channel_pt]; // the first row and column are invalid
			}

			// ReLU
			for (int channel_pt=0; channel_pt<BN_CHANNEL_PARALLELISM; channel_pt++) {
				if (out_feature_t0[channel_pt] < 0) out_feature_t0[channel_pt] *= relu_weight[channel_pt];
			}

			// Shortcut
			for (int channel_pt=0; channel_pt<BN_CHANNEL_PARALLELISM; channel_pt++) {
				out_feature_t1[channel_pt] += out_feature_t0[channel_pt];
			}

			// Batch Normalization and Write-back
			for (int channel_pt=0; channel_pt<BN_CHANNEL_PARALLELISM; channel_pt++) {
				residual[channel_tile][channel_pt][i][j] = bn_weight_1[channel_pt]*out_feature_t1[channel_pt] + bn_bias_1[channel_pt];
			}
		}
	}
}


/* 
Used for the shorcut in downsample layers 
To make sure its dimensionality matches with the residual outputs
*/
void avgpool_concat(
		FIX_FM_acc outputs[CHANNEL_OUT/CHANNEL_OUT_T][CHANNEL_OUT_T][WIDTH][WIDTH],
		int H_fmap,
		int in_channels
)
{
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=2

	int in_channel_blocks = in_channels/BN_CHANNEL_PARALLELISM;

	FIX_FM_acc out_feature[BN_CHANNEL_PARALLELISM];
#pragma HLS ARRAY_PARTITION variable=out_feature complete dim=1
	FIX_FM_acc out_tmp[BN_CHANNEL_PARALLELISM][WIDTH/2][WIDTH/2];
#pragma HLS ARRAY_PARTITION variable=out_tmp complete dim=1

	LOOP_init:
	for (int i = 0; i < WIDTH/2; i ++){
		for (int j = 0; j < WIDTH/2; j ++){
#pragma HLS PIPELINE
			for (int channel_pt = 0; channel_pt < BN_CHANNEL_PARALLELISM; channel_pt ++){
				out_tmp[channel_pt][i][j] = 0;
			}
		}
	}
	LOOP_avgpool:
	for (int tile=0; tile < in_channel_blocks; tile ++) {
		for (int i = 0; i < H_fmap; i ++){
			for (int j = 0; j < H_fmap; j ++){
				for (int ii = 0; ii < 2; ii ++){
					for (int jj = 0; jj < 2; jj ++){
#pragma HLS PIPELINE
						for (int channel_pt = 0; channel_pt < BN_CHANNEL_PARALLELISM; channel_pt ++){
							if (ii + jj == 0) {
								out_feature[channel_pt] = outputs[tile][channel_pt][i*2 + ii][j*2 + jj];;
							} else{
								out_feature[channel_pt] += outputs[tile][channel_pt][i*2 + ii][j*2 + jj];;
							}
						}
						for (int channel_pt = 0; channel_pt < BN_CHANNEL_PARALLELISM; channel_pt ++){
							out_tmp[channel_pt][i][j] = out_feature[channel_pt]/(FIX_FM_acc)4.0;
						}
					}
				}
			}
		}
		for (int i = 0; i < H_fmap; i ++){
			for (int j = 0; j < H_fmap; j ++){
#pragma HLS PIPELINE
				for (int channel_pt = 0; channel_pt < BN_CHANNEL_PARALLELISM; channel_pt ++){
					outputs[tile][channel_pt][i][j] = out_tmp[channel_pt][i][j];
					outputs[tile+in_channel_blocks][channel_pt][i][j] = out_tmp[channel_pt][i][j];
				}
			}
		}
	}

}

/* The final pooling layer before the FC layer */
void avgpool_8x8(
		FIX_FM_acc inputs[CHANNEL_OUT/CHANNEL_OUT_T][CHANNEL_OUT_T][WIDTH][WIDTH],
		FIX_32_10 outputs[CHANNEL_OUT]
)
{
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=2

	FIX_32_10 tmp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

	LOOP_avgpool_8x8:
	for (int tile = 0; tile < CHANNEL_OUT/OUT_CHANNEL_PARALLELISM; tile ++) {
		for (int k = 0; k < OUT_CHANNEL_PARALLELISM; k ++) {
#pragma HLS PIPELINE
			tmp[k] = 0;
		}
		for (int i = 0; i < 16; i ++) {
			for (int j = 0; j < 16; j ++) {
#pragma HLS PIPELINE
				for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
					tmp[ch] += inputs[tile][ch][i][j];
				}
			}
		}
		for (int k = 0; k < OUT_CHANNEL_PARALLELISM; k++) {
#pragma HLS PIPELINE
			outputs[tile*OUT_CHANNEL_PARALLELISM + k] = tmp[k]/(FIX_32_10)32.0;
		}
	}
}

/* FC layer */
void matmul(
		FIX_32_10 inputs[32],
		const FIX_WT linear_weight[2][32],
		const FIX_WT linear_bias[2],
		FIX_32_10 outputs[2]
)
{
#pragma HLS ARRAY_PARTITION variable=linear_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=linear_bias complete dim=1

	FIX_FM_acc buf[2];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1

	for(int coo = 0; coo < 2; coo ++) {
#pragma HLS PIPELINE
		buf[coo] = linear_bias[coo];
	}

	for(int cii = 0; cii < 32; cii++) {
#pragma HLS PIPELINE
		FIX_FM_acc tmp_in = inputs[cii];
		FIX_FM_acc tmp_wt[2];
#pragma HLS ARRAY_PARTITION variable=tmp_wt complete dim=1
		for(int coo = 0; coo < 2; coo ++) {
			tmp_wt[coo] = linear_weight[coo][cii];
		}
		for(int coo = 0; coo < 2; coo ++) {
			buf[coo] += tmp_in*tmp_wt[coo];
		}
	}

	for(int coo = 0; coo < 2; coo ++) {
#pragma HLS PIPELINE
		outputs[coo] = buf[coo];
	}
}

#endif

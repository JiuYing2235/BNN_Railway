#include "typedefs.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "bnn.h"
#include "weights_tb.h"
#include <fstream>
#include <hls_math.h>

using namespace std;

#define NUM_TESTS 88

unsigned char images[NUM_TESTS*96*32*32];
unsigned char labels[NUM_TESTS];

void load_image()
{
	std::ifstream ifs_param("conv1_input_my_0.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(images), sizeof(unsigned char)*96*NUM_TESTS*32*32);
	ifs_param.close();
}

void load_label()
{
	std::ifstream ifs_param("labels.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(labels), sizeof(unsigned char)*NUM_TESTS);
	ifs_param.close();
}

void get_image(unsigned char *images, unsigned int idx, float image[96][32][32])
{
	unsigned int offset = idx*96*32*32;
	for (int c = 0; c < 96; c ++) {
		for (int row = 0; row < 32; row ++) {
			for (int col = 0; col < 32; col ++) {
				image[c][row][col] = images[offset + c*32*32 + row*32 + col];
//				cout << image[c][row][col] << endl;
			}
		}
	}
}

//#define SW_TEST
int main(int argc, char **argv)
{
	int correct_sw = 0;
	int correct_hw = 0;

	// Generate the expected result
	// Iterate over the rows of the A matrix
	// int num_tests = 10;

	load_image();
	load_label();

	for (int k = 0; k < NUM_TESTS; k ++) {
		float image[96][32][32] = {0};
		float p;


		get_image(images, k, image);
//		for (int i = 0; i < 96; i++) {           // Assuming the first dimension size is 96
//		            for (int j = 0; j < 32; j++) {       // Assuming the second dimension size is 32
//		                for (int l = 0; l < 32; l++) {
//		cout << image[i][j][l] << endl;}}}


#ifdef LAYER_TEST
		int print_row = 8;
		int print_col = 8;

		cout << "tb output avg_pool_out" << endl;
//		for (int row = 0; row < print_row; row ++) {
//			for (int col = 0; col < print_col; col ++) {
//				cout << layer3_2_bn4_out[0][row][col] << "  ";
//			}
//			cout << endl;
//		}
		for (int i = 0; i < 64; i ++){
			cout << avg_pool_out[i];
		}
		cout << endl;
		cout << "-------------------- above is tb.cc output ---------------------------" << endl;
#endif



		////////////////////////////////
		//////// HARDWARE //////////////
		////////////////////////////////
#ifdef LAYER_TEST
		float accelerator_output[64*32*32];
#else
		float accelerator_output[2];
#endif

		uint64 image_hw[3][32][32] = {0};

		for(int j = 0; j < 3; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					for(int b = 0; b < 32; b ++){
						if (image[j*32 + b][row][col] > 0) {
							image_hw[j][row][col][63 - b] = 1;
						} else {
							image_hw[j][row][col][63 - b] = 0;
						}
					}
				}
			}
		}

		FracNet_T(image_hw, accelerator_output);

#ifdef LAYER_TEST
		cout << endl << "accelerator output: "<< endl;
//		for (int row = 0; row < print_row; row ++) {
//			for(int col = 0; col < print_col; col ++) {
//				cout << accelerator_output[row*32 + col] << "  ";
//			}
//			cout << endl;
//		}
		for (int i = 0; i < 64; i ++){
			cout << accelerator_output[i];
		}
		cout << endl;

		FIX_FM_acc err = 0;
		FIX_FM_acc total_err = 0;
		FIX_FM_acc max_err = 0;
		int err_cnt = 0;
		int total = 0;
		for(int i=0; i<1; i++){
			for(int j=0; j<1; j++){
				for(int k=0; k<64; k++){
					err = hls::absf(accelerator_output[i*32*32+j*32+k] - avg_pool_out[k]);
					if (err > max_err) max_err = err;
					if (err > 0.1) {
						err_cnt += 1;
						cout << "(" << i << ", " << j << ", " << k << ") " << endl;
					}
//					if (err != 0) cout << "(" << i << ", " << j << ", " << k << ") ";
					total_err += err;
					total += 1;
				}
			}
		}
		cout << endl << "Total absolute error: " << total_err << endl;
		cout << "Total number of errors: " << err_cnt << "/" << total << endl;
		cout << "Maximum absolute pixel error: " << max_err << endl;
#else

		int predict_hw = 0;
		int same = 0;
		p = -1000;
		for (int i = 0; i < 2; i ++) {
			float cl = accelerator_output[i];
			cout << accelerator_output[i] << "  ";
			if (cl > p) {
				p = cl;
				predict_hw = i;
			}
		}
		cout << endl;
		cout << predict_hw << endl;
		if (predict_hw == labels[k]) {
			correct_hw ++;
		}

		cout << "Hardware has "<< correct_hw << "/" << k + 1 << " correct." << endl;
		cout << "\n" << endl;
#endif

	}
	return 0;
}

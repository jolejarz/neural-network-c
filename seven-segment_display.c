// Jason W. Olejarz
//
// Building and Training of Artificial Neural Networks Using C
//
// In this example, consider a single digit of a seven-segment display. It has the following structure:
//
//            -----------
//           |     2     |
//           |           |
//           | 1       3 |
//           |           |
//           |     0     |
//            -----------
//           |           |
//           |           |
//           | 6       4 |
//           |           |
//           |     5     |
//            -----------
//
// Here, each of the seven display segments is labeled 0 through 7.
//
// Consider each of the sixteen hexadecimal digits based on which of the seven segments are lit:
//
//        0 = (1,1,1,1,1,1,0)     1 = (0,0,1,1,0,0,0)     2 = (1,1,0,1,1,0,1)     3 = (0,1,1,1,1,0,1)
//         binary: (0,0,0,0)       binary: (0,0,0,1)       binary: (0,0,1,0)       binary: (0,0,1,1)
//
//            -----------                                     -----------             ----------- 
//           |     2     |                 2     |                 2     |                 2     |
//           |           |                       |                       |                       |
//           | 1       3 |             1       3 |             1       3 |             1       3 |
//           |           |                       |                       |                       |
//           |     0     |                 0     |                 0     |                 0     |
//                                                            -----------             ----------- 
//           |           |                       |           |                                   |
//           |           |                       |           |                                   |
//           | 6       4 |             6       4 |           | 6       4               6       4 |
//           |           |                       |           |                                   |
//           |     5     |                 5     |           |     5                       5     |
//            -----------                                     -----------             ----------- 
//
//        4 = (0,0,1,1,0,1,1)     5 = (0,1,1,0,1,1,1)     6 = (1,1,1,0,1,1,1)     7 = (0,0,1,1,1,0,0)
//         binary: (0,1,0,0)       binary: (0,1,0,1)       binary: (0,1,1,0)       binary: (0,1,1,1)
//
//                                    -----------             -----------             ----------- 
//           |     2     |           |     2                 |     2                       2     |
//           |           |           |                       |                                   |
//           | 1       3 |           | 1       3             | 1       3               1       3 |
//           |           |           |                       |                                   |
//           |     0     |           |     0                 |     0                       0     |
//            -----------             -----------             -----------                         
//                       |                       |           |           |                       |
//                       |                       |           |           |                       |
//             6       4 |             6       4 |           | 6       4 |             6       4 |
//                       |                       |           |           |                       |
//                 5     |                 5     |           |     5     |                 5     |
//                                    -----------             -----------                         
//
//        8 = (1,1,1,1,1,1,1)     9 = (0,0,1,1,1,1,1)     A = (1,0,1,1,1,1,1)     B = (1,1,1,0,0,1,1)
//         binary: (1,0,0,0)       binary: (1,0,0,1)       binary: (1,0,1,0)       binary: (1,0,1,1)
//
//            -----------             -----------             -----------                         
//           |     2     |           |     2     |           |     2     |           |     2      
//           |           |           |           |           |           |           |            
//           | 1       3 |           | 1       3 |           | 1       3 |           | 1       3  
//           |           |           |           |           |           |           |            
//           |     0     |           |     0     |           |     0     |           |     0      
//            -----------             -----------             -----------             ----------- 
//           |           |                       |           |           |           |           |
//           |           |                       |           |           |           |           |
//           | 6       4 |             6       4 |           | 6       4 |           | 6       4 |
//           |           |                       |           |           |           |           |
//           |     5     |                 5     |           |     5     |           |     5     |
//            -----------                                                             ----------- 
//
//        C = (1,1,0,0,1,1,0)     D = (1,1,1,1,0,0,1)     E = (1,1,0,0,1,1,1)     F = (1,0,0,0,1,1,1)
//         binary: (1,1,0,0)       binary: (1,1,0,1)       binary: (1,1,1,0)       binary: (1,1,1,1)
//
//            -----------                                     -----------             ----------- 
//           |     2                       2     |           |     2                 |     2      
//           |                                   |           |                       |            
//           | 1       3               1       3 |           | 1       3             | 1       3  
//           |                                   |           |                       |            
//           |     0                       0     |           |     0                 |     0      
//                                    -----------             -----------             ----------- 
//           |                       |           |           |                       |            
//           |                       |           |           |                       |            
//           | 6       4             | 6       4 |           | 6       4             | 6       4  
//           |                       |           |           |                       |            
//           |     5                 |     5     |           |     5                 |     5      
//            -----------             -----------             -----------                         
//
// Each of the sixteen hexadecimal digits therefore has a seven-bit representation based on the seven-segment display.
//
// In this example, we build a feedforward neural network consisting of a 7-unit input layer, three 13-unit hidden layers, and one 5-unit output layer.
// The input is the seven-bit representation of each hexadecimal digit based on which portions of the seven-segment display are activated.
// The four low-order bits of the output are the binary representation of that digit.
// The high-order bit of the output is an indicator to signal if the seven-bit combination represents one of the sixteen hexadecimal digits.
// If this bit is 0, then the seven-bit combination corresponds to one of the hexadecimal digits.
// If this bit is 1, then the seven-bit combination does not correspond to one of the hexadecimal digits, and the low-order output bits are undefined.
//
// We train the network to produce the desired outputs. Its performance is displayed in the program's output.

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include "annl.h"

#define NUM 128
#define MIDDLE_1_SIZE 13
#define MIDDLE_2_SIZE 13
#define MIDDLE_3_SIZE 13

void status (int epoch, double loss);

int main (int argc, char *argv[])
{
	// These are loop indices.
	int i, j, n;

	// This is the name of the logic gate that the network is trained on.
	char gate[4];

	// Allocate memory for the inputs and outputs.
	char *hex_digit_all = malloc (sizeof(char)*NUM);
	double *input_all = malloc (sizeof(double)*NUM*7);
	double *output_all = malloc (sizeof(double)*NUM*5);
	double *output_target_all = malloc (sizeof(double)*NUM*5);
	double *output_target_fit_all = malloc (sizeof(double)*NUM*5);
	char *output_target_char_all = malloc (sizeof(char)*NUM*5);

	// Create the structures for the input layer, the middle layers, and the output layer.
	annlLayer *layer_input = annlCreateLayer (7, 0, NULL);
	annlLayer *layer_middle = annlCreateLayer (MIDDLE_1_SIZE, 1, annlActivateReLU);
	annlLayer *layer_middle_2 = annlCreateLayer (MIDDLE_2_SIZE, 1, annlActivateReLU);
	annlLayer *layer_middle_3 = annlCreateLayer (MIDDLE_3_SIZE, 1, annlActivateReLU);
	annlLayer *layer_output = annlCreateLayer (5, 1, annlActivateLogistic);

	// Specify the values of step and loss_diff.
	double step = 0.001;
	double loss_diff = 0.001;

	// Set up the pseudorandom number generator.
	gsl_rng *rng_mt = gsl_rng_alloc (gsl_rng_mt19937);
	gsl_rng_set (rng_mt, 1);

	// Set up the inputs and targeted outputs.
	double list[NUM][18] = {(double)' ', /*     0x00 input */ 0,0,0,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x01 input */ 0,0,0,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x02 input */ 0,0,0,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x03 input */ 0,0,0,0,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x04 input */ 0,0,0,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x05 input */ 0,0,0,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x06 input */ 0,0,0,0,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x07 input */ 0,0,0,0,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x08 input */ 0,0,0,1,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x09 input */ 0,0,0,1,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x0A input */ 0,0,0,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x0B input */ 0,0,0,1,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x0C input */ 0,0,0,1,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x0D input */ 0,0,0,1,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x0E input */ 0,0,0,1,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x0F input */ 0,0,0,1,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x10 input */ 0,0,1,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x11 input */ 0,0,1,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x12 input */ 0,0,1,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x13 input */ 0,0,1,0,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x14 input */ 0,0,1,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x15 input */ 0,0,1,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x16 input */ 0,0,1,0,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x17 input */ 0,0,1,0,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'1', /* '1' 0x18 input */ 0,0,1,1,0,0,0, /* output_target */ 0,0,0,0,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x19 input */ 0,0,1,1,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x1A input */ 0,0,1,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'4', /* '4' 0x1B input */ 0,0,1,1,0,1,1, /* output_target */ 0,0,1,0,0, /* output_target_fit */ 1,1,1,1,1,
				(double)'7', /* '7' 0x1C input */ 0,0,1,1,1,0,0, /* output_target */ 0,0,1,1,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x1D input */ 0,0,1,1,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x1E input */ 0,0,1,1,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'9', /* '9' 0x1F input */ 0,0,1,1,1,1,1, /* output_target */ 0,1,0,0,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x20 input */ 0,1,0,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x21 input */ 0,1,0,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x22 input */ 0,1,0,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x23 input */ 0,1,0,0,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x24 input */ 0,1,0,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x25 input */ 0,1,0,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x26 input */ 0,1,0,0,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x27 input */ 0,1,0,0,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x28 input */ 0,1,0,1,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x29 input */ 0,1,0,1,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x2A input */ 0,1,0,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x2B input */ 0,1,0,1,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x2C input */ 0,1,0,1,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x2D input */ 0,1,0,1,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x2E input */ 0,1,0,1,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x2F input */ 0,1,0,1,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x30 input */ 0,1,1,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x31 input */ 0,1,1,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x32 input */ 0,1,1,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x33 input */ 0,1,1,0,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x34 input */ 0,1,1,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x35 input */ 0,1,1,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x36 input */ 0,1,1,0,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'5', /* '5' 0x37 input */ 0,1,1,0,1,1,1, /* output_target */ 0,0,1,0,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x38 input */ 0,1,1,1,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x39 input */ 0,1,1,1,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x3A input */ 0,1,1,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x3B input */ 0,1,1,1,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x3C input */ 0,1,1,1,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'3', /* '3' 0x3D input */ 0,1,1,1,1,0,1, /* output_target */ 0,0,0,1,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x3E input */ 0,1,1,1,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x3F input */ 1,0,0,1,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x40 input */ 1,0,0,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x41 input */ 1,0,0,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x42 input */ 1,0,0,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x43 input */ 1,0,0,0,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x44 input */ 1,0,0,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x45 input */ 1,0,0,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x46 input */ 1,0,0,0,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'F', /* 'F' 0x47 input */ 1,0,0,0,1,1,1, /* output_target */ 0,1,1,1,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x48 input */ 1,0,0,1,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x49 input */ 1,0,0,1,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x4A input */ 1,0,0,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x4B input */ 1,0,0,1,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x4C input */ 1,0,0,1,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x4D input */ 1,0,0,1,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x4E input */ 1,0,0,1,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x4F input */ 1,0,0,1,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x50 input */ 1,0,1,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x51 input */ 1,0,1,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x52 input */ 1,0,1,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x53 input */ 1,0,1,0,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x54 input */ 1,0,1,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x55 input */ 1,0,1,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x56 input */ 1,0,1,0,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x57 input */ 1,0,1,0,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x58 input */ 1,0,1,1,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x59 input */ 1,0,1,1,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x5A input */ 1,0,1,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x5B input */ 1,0,1,1,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x5C input */ 1,0,1,1,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x5D input */ 1,0,1,1,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x5E input */ 1,0,1,1,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'A', /* 'A' 0x5F input */ 1,0,1,1,1,1,1, /* output_target */ 0,1,0,1,0, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x60 input */ 1,1,0,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x61 input */ 1,1,0,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x62 input */ 1,1,0,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x63 input */ 1,1,0,0,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x64 input */ 1,1,0,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x65 input */ 1,1,0,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'C', /* 'C' 0x66 input */ 1,1,0,0,1,1,0, /* output_target */ 0,1,1,0,0, /* output_target_fit */ 1,1,1,1,1,
				(double)'E', /* 'E' 0x67 input */ 1,1,0,0,1,1,1, /* output_target */ 0,1,1,1,0, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x68 input */ 1,1,0,1,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x69 input */ 1,1,0,1,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x6A input */ 1,1,0,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x6B input */ 1,1,0,1,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x6C input */ 1,1,0,1,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'2', /* '2' 0x6D input */ 1,1,0,1,1,0,1, /* output_target */ 0,0,0,1,0, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x6E input */ 1,1,0,1,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x6F input */ 1,1,0,1,1,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x70 input */ 1,1,1,0,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x71 input */ 1,1,1,0,0,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x72 input */ 1,1,1,0,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'B', /* 'B' 0x73 input */ 1,1,1,0,0,1,1, /* output_target */ 0,1,0,1,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x74 input */ 1,1,1,0,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x75 input */ 1,1,1,0,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x76 input */ 1,1,1,0,1,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'6', /* '6' 0x77 input */ 1,1,1,0,1,1,1, /* output_target */ 0,0,1,1,0, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x78 input */ 1,1,1,1,0,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'D', /* 'D' 0x79 input */ 1,1,1,1,0,0,1, /* output_target */ 0,1,1,0,1, /* output_target_fit */ 1,1,1,1,1,
				(double)' ', /*     0x7A input */ 1,1,1,1,0,1,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x7B input */ 1,1,1,1,0,1,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x7C input */ 1,1,1,1,1,0,0, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)' ', /*     0x7D input */ 1,1,1,1,1,0,1, /* output_target */ 1,0,0,0,0, /* output_target_fit */ 1,0,0,0,0,
				(double)'0', /* '0' 0x7E input */ 1,1,1,1,1,1,0, /* output_target */ 0,0,0,0,0, /* output_target_fit */ 1,1,1,1,1,
				(double)'8', /* '8' 0x7F input */ 1,1,1,1,1,1,1, /* output_target */ 0,1,0,0,0, /* output_target_fit */ 1,1,1,1,1};

	// Set the input and targeted output arrays.
	for (i=0; i<NUM; i++)
	{
		*(hex_digit_all+i) = (char)list[i][0];
		for (j=0; j<7; j++)
		{
			*(input_all+7*i+j) = list[i][1+j];
		}
		for (j=0; j<5; j++)
		{
			*(output_target_all+5*i+j) = list[i][8+j];
			*(output_target_fit_all+5*i+j) = list[i][13+j];

			if (j==0 || list[i][8]==0) *(output_target_char_all+5*i+j) = list[i][8+j] == 0 ? '0' : '1';
			else *(output_target_char_all+5*i+j) = 'X';
		}
	}

	// Set the sequence.
	annlLinkSequence (layer_input, layer_middle);
	annlLinkSequence (layer_middle, layer_middle_2);
	annlLinkSequence (layer_middle_2, layer_middle_3);
	annlLinkSequence (layer_middle_3, layer_output);

	// Set up the biases.
	annlSetBiasFull (layer_middle, TRAIN_ADAM);
	annlSetBiasFull (layer_middle_2, TRAIN_ADAM);
	annlSetBiasFull (layer_middle_3, TRAIN_ADAM);
	annlSetBiasFull (layer_output, TRAIN_ADAM);

	// Connect the layers.
	annlConnectFull (layer_input, layer_middle, TRAIN_ADAM);
	annlConnectFull (layer_middle, layer_middle_2, TRAIN_ADAM);
	annlConnectFull (layer_middle_2, layer_middle_3, TRAIN_ADAM);
	annlConnectFull (layer_middle_3, layer_output, TRAIN_ADAM);

	// Set the initial weights and biases.
	annlRandomizeParameters (layer_middle, rng_mt);
	annlRandomizeParameters (layer_middle_2, rng_mt);
	annlRandomizeParameters (layer_middle_3, rng_mt);
	annlRandomizeParameters (layer_output, rng_mt);

	annlSequence sequence;
	annlSequenceList sequence_list[NUM];

	sequence.num_sequence = NUM;
	sequence.sequence_list = sequence_list;

	annlSequenceInput sequence_input[NUM];
	annlSequenceOutput sequence_output[NUM];

	for (int i=0; i<NUM; i++)
	{
		sequence.sequence_list[i].layer_start = layer_input;
		sequence.sequence_list[i].num_layer_input = 1;
		sequence.sequence_list[i].num_layer_output = 1;
		sequence.sequence_list[i].layer_input_list = &sequence_input[i];
		sequence.sequence_list[i].layer_output_list = &sequence_output[i];

		sequence_input[i].layer_input = layer_input;
		sequence_input[i].input_values = &list[i][1];

		sequence_output[i].layer_output = layer_output;
		sequence_output[i].output_values = &output_all[5*i];
		sequence_output[i].output_target = &list[i][8];
		sequence_output[i].output_target_fit = &list[i][13];
	}

	// Train the network.
	annlTrain (sequence, layer_input, annlCalculateLossSquaredError, loss_diff, 128, NULL, step, status);

	// Print the outputs.
	for (i=0; i<NUM; i++)
	{
		printf ("Hex digit = %c; Input = (%d,%d,%d,%d,%d,%d,%d); θ[Output-1/2] = (%d,%d,%d,%d,%d); Target = (%c,%c,%c,%c,%c); Output = (%lf,%lf,%lf,%lf,%lf)\n",
		        *(hex_digit_all+i),
			(int)(*(input_all+7*i+0)),
			(int)(*(input_all+7*i+1)),
			(int)(*(input_all+7*i+2)),
			(int)(*(input_all+7*i+3)),
			(int)(*(input_all+7*i+4)),
			(int)(*(input_all+7*i+5)),
			(int)(*(input_all+7*i+6)),
			annlHeavisideTheta(*(output_all+5*i+0)-0.5),
			annlHeavisideTheta(*(output_all+5*i+1)-0.5),
			annlHeavisideTheta(*(output_all+5*i+2)-0.5),
			annlHeavisideTheta(*(output_all+5*i+3)-0.5),
			annlHeavisideTheta(*(output_all+5*i+4)-0.5),
			*(output_target_char_all+5*i+0),
			*(output_target_char_all+5*i+1),
			*(output_target_char_all+5*i+2),
			*(output_target_char_all+5*i+3),
			*(output_target_char_all+5*i+4),
			*(output_all+5*i+0),
			*(output_all+5*i+1),
			*(output_all+5*i+2),
			*(output_all+5*i+3),
			*(output_all+5*i+4));
	}

	return 0;
}

void status (int epoch, double loss)
{
	printf("Epoch = %d, Loss = %lf\n", epoch, loss);
	return;
}

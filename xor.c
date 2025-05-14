// Jason W. Olejarz
//
// Building and Training of Artificial Neural Networks Using C

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include "annl.h"

#define NUM 4
#define MIDDLE_SIZE 3

int main (int argc, char *argv[])
{
	// These are loop indices.
	int i, j, n;

	// This is the name of the logic gate that the network is trained on.
	char gate[4];

	// Allocate memory for the inputs and outputs.
	double *input_all = malloc (sizeof(double)*NUM*2);
	double *output_all = malloc (sizeof(double)*NUM*1);
	double *output_target_all = malloc (sizeof(double)*NUM*1);
	double *output_target_fit_all = malloc (sizeof(double)*NUM*1);

	// Create the structures for the input layer, the middle layers, and the output layer.
	annlLayer *layer_input = annlCreateLayer (2, 0, NULL);
	annlLayer *layer_middle = annlCreateLayer (3, 1, annlActivateReLU);
	annlLayer *layer_output = annlCreateLayer (1, 1, annlActivateLogistic);

	// Specify the values of step and loss_diff.
	double step = 0.001;
	double loss_diff = 0.001;

	// Set up the pseudorandom number generator.
	gsl_rng *rng_mt = gsl_rng_alloc (gsl_rng_mt19937);
	gsl_rng_set (rng_mt, 1);

	// Set up the inputs and targeted outputs.
	double list[NUM][5] = {(double)' ', /* 0x00 input */ 0,0, /* output_target */ 0, /* output_target_fit */ 1,
	      		       (double)' ', /* 0x01 input */ 0,1, /* output_target */ 1, /* output_target_fit */ 1,
			       (double)' ', /* 0x02 input */ 1,0, /* output_target */ 1, /* output_target_fit */ 1,
			       (double)' ', /* 0x03 input */ 1,1, /* output_target */ 0, /* output_target_fit */ 1};

	// Set the input and targeted output arrays.
	for (i=0; i<NUM; i++)
	{
		for (j=0; j<2; j++)
		{
			*(input_all+2*i+j) = list[i][1+j];
		}
		for (j=0; j<1; j++)
		{
			*(output_target_all+1*i+j) = list[i][3+j];
			*(output_target_fit_all+1*i+j) = list[i][4+j];
		}
	}

	// Set the sequence.
	annlLinkSequence (layer_input, layer_middle);
	annlLinkSequence (layer_middle, layer_output);

	// Set up the biases.
	annlSetBiasFull (layer_middle, TRAIN_ADAM);
	annlSetBiasFull (layer_output, TRAIN_ADAM);

	// Connect the layers.
	annlConnectFull (layer_input, layer_middle, TRAIN_ADAM);
	annlConnectFull (layer_middle, layer_output, TRAIN_ADAM);

	// Set the initial weights and biases.
	annlRandomizeParameters (layer_middle, rng_mt);
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
		sequence_output[i].output_values = &output_all[i];
		sequence_output[i].output_target = &list[i][3];
		sequence_output[i].output_target_fit = &list[i][4];
	}

	int epoch = 0;
	double loss_test;

	// Check if the total loss is greater than loss_diff.
	while ( (loss_test=annlCalculateLossTotal (sequence)) > loss_diff )
	{
		printf("Epoch = %d, Loss = %lf\n", epoch++, loss_test);

		// Calculate the gradient.
		annlCalculateGradient (sequence);

		// Update the parameters.
		annlUpdateParameters (layer_input, step);
	}

	// Print the outputs.
	for (i=0; i<NUM; i++)
	{
		printf ("Input = (%d,%d); θ[Output-1/2] = (%d); Output = (%lf)\n", (int)(*(input_all+2*i+0)), (int)(*(input_all+2*i+1)), annlHeavisideTheta(*(output_all+1*i+0)-0.5), *(output_all+1*i+0));
	}

	return 0;
}

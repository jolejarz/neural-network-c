// Jason W. Olejarz
//
// Building and Training of Artificial Neural Networks Using C

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include "annl.h"

#define NUM 1

void status (int epoch, double loss);

int main (int argc, char *argv[])
{
	// These are loop indices.
	int i, j, n;

	// This is the name of the logic gate that the network is trained on.
	char gate[4];

	// Allocate memory for the inputs and outputs.
	double *input_all = malloc (sizeof(double)*NUM*16);
	double *output_all = malloc (sizeof(double)*NUM*16);
	double *output_target_all = malloc (sizeof(double)*NUM*16);
	double *output_target_fit_all = malloc (sizeof(double)*NUM*16);

	// Create the structures for the input layer, the middle layers, and the output layer.
	annlLayer *layer_input_0 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_0 = annlCreateLayer (2, 1, annlActivateReLU);
	annlLayer *layer_output_0 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_1 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_1 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_1 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_2 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_2 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_2 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_3 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_3 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_3 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_4 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_4 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_4 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_5 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_5 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_5 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_6 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_6 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_6 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_7 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_7 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_7 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_8 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_8 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_8 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_9 = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_9 = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_9 = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_A = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_A = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_A = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_B = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_B = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_B = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_C = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_C = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_C = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_D = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_D = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_D = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_E = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_E = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_E = annlCreateLayer (1, 1, annlActivateLogistic);
	annlLayer *layer_input_F = annlCreateLayer (1, 0, NULL);
	annlLayer *layer_middle_F = annlCreateLayer (2, 2, annlActivateReLU);
	annlLayer *layer_output_F = annlCreateLayer (1, 1, annlActivateLogistic);

	// Specify the values of step and loss_diff.
	double step = 0.001;
	double loss_diff = 0.001;

	// Set up the pseudorandom number generator.
	gsl_rng *rng_mt = gsl_rng_alloc (gsl_rng_mt19937);
	gsl_rng_set (rng_mt, 1);

	// Set up the inputs and targeted outputs.
	double list[NUM][49] = {(double)' ', /* input */ 0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,
		                     /* output_target */ 0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,
			         /* output_target_fit */ 0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

	// Set the input and targeted output arrays.
	for (i=0; i<NUM; i++)
	{
		for (j=0; j<16; j++)
		{
			*(input_all+1*i+j) = list[i][1+j];
		}
		for (j=0; j<16; j++)
		{
			*(output_target_all+1*i+j) = list[i][17+j];
			*(output_target_fit_all+1*i+j) = list[i][33+j];
		}
	}

	// Set the sequence.
	annlLinkSequence (layer_input_0, layer_middle_0);
	annlLinkSequence (layer_middle_0, layer_output_0);
	annlLinkSequence (layer_output_0, layer_middle_1);
	annlLinkSequence (layer_middle_1, layer_output_1);
	annlLinkSequence (layer_output_1, layer_middle_2);
	annlLinkSequence (layer_middle_2, layer_output_2);
	annlLinkSequence (layer_output_2, layer_middle_3);
	annlLinkSequence (layer_middle_3, layer_output_3);
	annlLinkSequence (layer_output_3, layer_middle_4);
	annlLinkSequence (layer_middle_4, layer_output_4);
	annlLinkSequence (layer_output_4, layer_middle_5);
	annlLinkSequence (layer_middle_5, layer_output_5);
	annlLinkSequence (layer_output_5, layer_middle_6);
	annlLinkSequence (layer_middle_6, layer_output_6);
	annlLinkSequence (layer_output_6, layer_middle_7);
	annlLinkSequence (layer_middle_7, layer_output_7);
	annlLinkSequence (layer_output_7, layer_middle_8);
	annlLinkSequence (layer_middle_8, layer_output_8);
	annlLinkSequence (layer_output_8, layer_middle_9);
	annlLinkSequence (layer_middle_9, layer_output_9);
	annlLinkSequence (layer_output_9, layer_middle_A);
	annlLinkSequence (layer_middle_A, layer_output_A);
	annlLinkSequence (layer_output_A, layer_middle_B);
	annlLinkSequence (layer_middle_B, layer_output_B);
	annlLinkSequence (layer_output_B, layer_middle_C);
	annlLinkSequence (layer_middle_C, layer_output_C);
	annlLinkSequence (layer_output_C, layer_middle_D);
	annlLinkSequence (layer_middle_D, layer_output_D);
	annlLinkSequence (layer_output_D, layer_middle_E);
	annlLinkSequence (layer_middle_E, layer_output_E);
	annlLinkSequence (layer_output_E, layer_middle_F);
	annlLinkSequence (layer_middle_F, layer_output_F);

	// Set up the biases.
	annlSetBiasFull (layer_middle_0, TRAIN_ADAM);
	annlSetBiasFull (layer_output_0, TRAIN_ADAM);
	annlSetBiasFullExisting (layer_middle_1, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_1, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_2, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_2, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_3, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_3, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_4, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_4, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_5, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_5, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_6, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_6, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_7, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_7, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_8, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_8, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_9, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_9, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_A, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_A, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_B, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_B, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_C, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_C, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_D, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_D, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_E, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_E, layer_output_0->b, layer_output_0->db);
	annlSetBiasFullExisting (layer_middle_F, layer_middle_0->b, layer_middle_0->db);
	annlSetBiasFullExisting (layer_output_F, layer_output_0->b, layer_output_0->db);

	// Connect the layers.
	annlConnectFull (layer_input_0, layer_middle_0, TRAIN_ADAM);
	annlConnectFull (layer_middle_0, layer_output_0, TRAIN_ADAM);
	annlConnectFullExisting (layer_input_1, layer_middle_1, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFull (layer_middle_0, layer_middle_1, TRAIN_ADAM);
	annlConnectFullExisting (layer_middle_1, layer_output_1, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_2, layer_middle_2, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_1, layer_middle_2, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_2, layer_output_2, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_3, layer_middle_3, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_2, layer_middle_3, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_3, layer_output_3, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_4, layer_middle_4, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_3, layer_middle_4, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_4, layer_output_4, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_5, layer_middle_5, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_4, layer_middle_5, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_5, layer_output_5, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_6, layer_middle_6, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_5, layer_middle_6, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_6, layer_output_6, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_7, layer_middle_7, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_6, layer_middle_7, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_7, layer_output_7, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_8, layer_middle_8, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_7, layer_middle_8, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_8, layer_output_8, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_9, layer_middle_9, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_8, layer_middle_9, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_9, layer_output_9, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_A, layer_middle_A, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_9, layer_middle_A, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_A, layer_output_A, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_B, layer_middle_B, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_A, layer_middle_B, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_B, layer_output_B, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_C, layer_middle_C, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_B, layer_middle_C, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_C, layer_output_C, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_D, layer_middle_D, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_C, layer_middle_D, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_D, layer_output_D, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_E, layer_middle_E, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_D, layer_middle_E, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_E, layer_output_E, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);
	annlConnectFullExisting (layer_input_F, layer_middle_F, layer_middle_0->layer_w[0].w, layer_middle_0->layer_w[0].dw);
	annlConnectFullExisting (layer_middle_E, layer_middle_F, layer_middle_1->layer_w[1].w, layer_middle_1->layer_w[1].dw);
	annlConnectFullExisting (layer_middle_F, layer_output_F, layer_output_0->layer_w[0].w, layer_output_0->layer_w[0].dw);

	// Set the initial weights and biases.
	annlRandomizeParameters (layer_middle_1, rng_mt);
	annlRandomizeParameters (layer_output_1, rng_mt);

	annlSequence sequence;
	annlSequenceList sequence_list[NUM];

	sequence.num_sequence = NUM;
	sequence.sequence_list = sequence_list;

	annlSequenceInput sequence_input[16];
	annlSequenceOutput sequence_output[16];

	for (int i=0; i<NUM; i++)
	{
		sequence.sequence_list[i].layer_start = layer_input_0;
		sequence.sequence_list[i].num_layer_input = 16;
		sequence.sequence_list[i].num_layer_output = 16;
		sequence.sequence_list[i].layer_input_list = &sequence_input[i];
		sequence.sequence_list[i].layer_output_list = &sequence_output[i];

		sequence_input[0].layer_input = layer_input_0;
		sequence_input[0].input_values = &list[i][1];
		sequence_input[1].layer_input = layer_input_1;
		sequence_input[1].input_values = &list[i][2];
		sequence_input[2].layer_input = layer_input_2;
		sequence_input[2].input_values = &list[i][3];
		sequence_input[3].layer_input = layer_input_3;
		sequence_input[3].input_values = &list[i][4];
		sequence_input[4].layer_input = layer_input_4;
		sequence_input[4].input_values = &list[i][5];
		sequence_input[5].layer_input = layer_input_5;
		sequence_input[5].input_values = &list[i][6];
		sequence_input[6].layer_input = layer_input_6;
		sequence_input[6].input_values = &list[i][7];
		sequence_input[7].layer_input = layer_input_7;
		sequence_input[7].input_values = &list[i][8];
		sequence_input[8].layer_input = layer_input_8;
		sequence_input[8].input_values = &list[i][9];
		sequence_input[9].layer_input = layer_input_9;
		sequence_input[9].input_values = &list[i][10];
		sequence_input[10].layer_input = layer_input_A;
		sequence_input[10].input_values = &list[i][11];
		sequence_input[11].layer_input = layer_input_B;
		sequence_input[11].input_values = &list[i][12];
		sequence_input[12].layer_input = layer_input_C;
		sequence_input[12].input_values = &list[i][13];
		sequence_input[13].layer_input = layer_input_D;
		sequence_input[13].input_values = &list[i][14];
		sequence_input[14].layer_input = layer_input_E;
		sequence_input[14].input_values = &list[i][15];
		sequence_input[15].layer_input = layer_input_F;
		sequence_input[15].input_values = &list[i][16];

		sequence_output[0].layer_output = layer_output_0;
		sequence_output[0].output_values = &output_all[0];
		sequence_output[0].output_target = &list[i][17];
		sequence_output[0].output_target_fit = &list[i][33];
		sequence_output[1].layer_output = layer_output_1;
		sequence_output[1].output_values = &output_all[1];
		sequence_output[1].output_target = &list[i][18];
		sequence_output[1].output_target_fit = &list[i][34];
		sequence_output[2].layer_output = layer_output_2;
		sequence_output[2].output_values = &output_all[2];
		sequence_output[2].output_target = &list[i][19];
		sequence_output[2].output_target_fit = &list[i][35];
		sequence_output[3].layer_output = layer_output_3;
		sequence_output[3].output_values = &output_all[3];
		sequence_output[3].output_target = &list[i][20];
		sequence_output[3].output_target_fit = &list[i][36];
		sequence_output[4].layer_output = layer_output_4;
		sequence_output[4].output_values = &output_all[4];
		sequence_output[4].output_target = &list[i][21];
		sequence_output[4].output_target_fit = &list[i][37];
		sequence_output[5].layer_output = layer_output_5;
		sequence_output[5].output_values = &output_all[5];
		sequence_output[5].output_target = &list[i][22];
		sequence_output[5].output_target_fit = &list[i][38];
		sequence_output[6].layer_output = layer_output_6;
		sequence_output[6].output_values = &output_all[6];
		sequence_output[6].output_target = &list[i][23];
		sequence_output[6].output_target_fit = &list[i][39];
		sequence_output[7].layer_output = layer_output_7;
		sequence_output[7].output_values = &output_all[7];
		sequence_output[7].output_target = &list[i][24];
		sequence_output[7].output_target_fit = &list[i][40];
		sequence_output[8].layer_output = layer_output_8;
		sequence_output[8].output_values = &output_all[8];
		sequence_output[8].output_target = &list[i][25];
		sequence_output[8].output_target_fit = &list[i][41];
		sequence_output[9].layer_output = layer_output_9;
		sequence_output[9].output_values = &output_all[9];
		sequence_output[9].output_target = &list[i][26];
		sequence_output[9].output_target_fit = &list[i][42];
		sequence_output[10].layer_output = layer_output_A;
		sequence_output[10].output_values = &output_all[10];
		sequence_output[10].output_target = &list[i][27];
		sequence_output[10].output_target_fit = &list[i][43];
		sequence_output[11].layer_output = layer_output_B;
		sequence_output[11].output_values = &output_all[11];
		sequence_output[11].output_target = &list[i][28];
		sequence_output[11].output_target_fit = &list[i][44];
		sequence_output[12].layer_output = layer_output_C;
		sequence_output[12].output_values = &output_all[12];
		sequence_output[12].output_target = &list[i][29];
		sequence_output[12].output_target_fit = &list[i][45];
		sequence_output[13].layer_output = layer_output_D;
		sequence_output[13].output_values = &output_all[13];
		sequence_output[13].output_target = &list[i][30];
		sequence_output[13].output_target_fit = &list[i][46];
		sequence_output[14].layer_output = layer_output_E;
		sequence_output[14].output_values = &output_all[14];
		sequence_output[14].output_target = &list[i][31];
		sequence_output[14].output_target_fit = &list[i][47];
		sequence_output[15].layer_output = layer_output_F;
		sequence_output[15].output_values = &output_all[15];
		sequence_output[15].output_target = &list[i][32];
		sequence_output[15].output_target_fit = &list[i][48];
	}

	// Train the network.
	annlTrain (sequence, layer_input_0, loss_diff, 1, NULL, step, status);

	// Print the outputs.
	for (i=0; i<NUM; i++)
	{
		printf ("Input = (%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d); θ[Output-1/2] = (%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d); Output = (%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf)\n",
		        (int)(*(input_all+1*i+0)),
			(int)(*(input_all+1*i+1)),
			(int)(*(input_all+1*i+2)),
			(int)(*(input_all+1*i+3)),
			(int)(*(input_all+1*i+4)),
			(int)(*(input_all+1*i+5)),
			(int)(*(input_all+1*i+6)),
			(int)(*(input_all+1*i+7)),
			(int)(*(input_all+1*i+8)),
			(int)(*(input_all+1*i+9)),
			(int)(*(input_all+1*i+10)),
			(int)(*(input_all+1*i+11)),
			(int)(*(input_all+1*i+12)),
			(int)(*(input_all+1*i+13)),
			(int)(*(input_all+1*i+14)),
			(int)(*(input_all+1*i+15)),
			annlHeavisideTheta(*(output_all+1*i+0)-0.5),
			annlHeavisideTheta(*(output_all+1*i+1)-0.5),
			annlHeavisideTheta(*(output_all+1*i+2)-0.5),
			annlHeavisideTheta(*(output_all+1*i+3)-0.5),
			annlHeavisideTheta(*(output_all+1*i+4)-0.5),
			annlHeavisideTheta(*(output_all+1*i+5)-0.5),
			annlHeavisideTheta(*(output_all+1*i+6)-0.5),
			annlHeavisideTheta(*(output_all+1*i+7)-0.5),
			annlHeavisideTheta(*(output_all+1*i+8)-0.5),
			annlHeavisideTheta(*(output_all+1*i+9)-0.5),
			annlHeavisideTheta(*(output_all+1*i+10)-0.5),
			annlHeavisideTheta(*(output_all+1*i+11)-0.5),
			annlHeavisideTheta(*(output_all+1*i+12)-0.5),
			annlHeavisideTheta(*(output_all+1*i+13)-0.5),
			annlHeavisideTheta(*(output_all+1*i+14)-0.5),
			annlHeavisideTheta(*(output_all+1*i+15)-0.5),
			*(output_all+1*i+0),
			*(output_all+1*i+1),
			*(output_all+1*i+2),
			*(output_all+1*i+3),
			*(output_all+1*i+4),
			*(output_all+1*i+5),
			*(output_all+1*i+6),
			*(output_all+1*i+7),
			*(output_all+1*i+8),
			*(output_all+1*i+9),
			*(output_all+1*i+10),
			*(output_all+1*i+11),
			*(output_all+1*i+12),
			*(output_all+1*i+13),
			*(output_all+1*i+14),
			*(output_all+1*i+15));
	}

	return 0;
}

void status (int epoch, double loss)
{
	printf("Epoch = %d, Loss = %lf\n", epoch, loss);
	return;
}

#include <gsl/gsl_rng.h>
#include <omp.h>

#define NO_DERIVATIVE 0
#define DERIVATIVE 1

#define TRAIN_BASIC 0
#define TRAIN_ADAM 1

// This is the index structure for the weight parameters.
// It is used for convolutional and pooling layers.
typedef struct
{
	int wij_index;
	int xi_index;
	int xj_index;
	int wij_list_previous;
	int wij_list_next;
	int xi_list_previous;
	int xi_list_next;
	int xj_list_previous;
	int xj_list_next;
} annlIndex_w;

// This is the index structure for the bias parameters.
// It is used for convolutional and pooling layers.
typedef struct
{
	int bi_index;
	int xi_index;
	int bi_list_previous;
	int bi_list_next;
	int xi_list_previous;
	int xi_list_next;
} annlIndex_b;

// This is the structure for the weight parameters.
typedef struct
{
	void *layer;
	int w_num;
	double *w;
	double *dw;
	double *w_m;
	double *w_v;
	int update_w;
	void (*calc_z_w)(void*,int);
	void (*calc_dw)(void*,int);
	void (*calc_dxj)(void*,int);
	void (*integrate_dw)(void*,int,double);
	double *w_parameters_trainable; // Used for convolutional and pooling layers
	int *wij_list_start_trainable; // Used for convolutional and pooling layers
	int *wij_xj_list_start_trainable; // Used for convolutional and pooling layers
	int *wij_xi_list_start_trainable; // Used for convolutional and pooling layers
	annlIndex_w *wij_list_trainable; // Used for convolutional and pooling layers
	double *w_parameters_not_trainable; // Used for convolutional and pooling layers
	int *wij_list_start_not_trainable; // Used for convolutional and pooling layers
	int *wij_xj_list_start_not_trainable; // Used for convolutional and pooling layers
	int *wij_xi_list_start_not_trainable; // Used for convolutional and pooling layers
	annlIndex_w *wij_list_not_trainable; // Used for convolutional and pooling layers
	double *w_value; // Used for convolutional and pooling layers
	int w_xj_num; // Used for convolutional and pooling layers
	int w_xi_num; // Used for convolutional and pooling layers
	int *w_update; // Used for fully connected layers
	int *w_index; // Used for fully connected layers
} annlLayer_w;

// This is the layer structure.
typedef struct
{
	int size;
	void *layer_previous;
	void *layer_next;
	int num_layer_w;
	annlLayer_w *layer_w;
	void (*calc_z_b)(void*);
	void (*calc_db)(void*);
	void (*integrate_db)(void*,double);
	void (*activation)(void*,int);
	double *z;
	double *dz;
	double *x;
	double *dx;
	double *b;
	double *db;
	double *b_m;
	double *b_v;
	int update_b;
	int b_num;
	double *b_parameters_trainable; // Used for convolutional and pooling layers
	int *bi_list_start_trainable; // Used for convolutional and pooling layers
	int *bi_xi_list_start_trainable; // Used for convolutional and pooling layers
	annlIndex_b *bi_list_trainable; // Used for convolutional and pooling layers
	double *b_parameters_not_trainable; // Used for convolutional and pooling layers
	int *bi_list_start_not_trainable; // Used for convolutional and pooling layers
	int *bi_xi_list_start_not_trainable; // Used for convolutional and pooling layers
	annlIndex_b *bi_list_not_trainable; // Used for convolutional and pooling layers
	double *b_value; // Used for convolutional and pooling layers
	int b_xi_num; // Used for convolutional and pooling layers
	int *b_update; // Used for fully connected layers
	int *b_index; // Used for fully connected layers
} annlLayer;

// This is the structure for the targeted outputs.
typedef struct
{
	annlLayer *layer_output;
	double *output_values;
	double *output_target;
	double *output_target_fit;
} annlSequenceOutput;

// This is the structure for the inputs.
typedef struct
{
	annlLayer *layer_input;
	double *input_values;
} annlSequenceInput;

// This is the sequence list structure.
typedef struct
{
	annlLayer *layer_start;
	int num_layer_input;
	int num_layer_output;
	annlSequenceInput *layer_input_list;
	annlSequenceOutput *layer_output_list;
} annlSequenceList;

// This is the sequence structure.
typedef struct
{
	int num_sequence;
	annlSequenceList *sequence_list;
} annlSequence;

// activation.c
void annlActivateReLU (annlLayer *layer_current, int derivative);
void annlActivateLogistic (annlLayer *layer_current, int derivative);
void annlActivateTanh (annlLayer *layer_current, int derivative);
void annlActivateSoftmax (annlLayer *layer_current, int derivative);
int annlHeavisideTheta (double x);

// bias.c
void annlSetBiasFull (annlLayer *layer_current, int train);
void annlSetBiasFullExisting (annlLayer *layer_current, double *b, double *db);
void annlSetBiasFullExisting_b (annlLayer *layer_current, double *b);
void annlSetBiasConvolution (annlLayer *layer_current, int L, int n, int train);
void annlSetBiasConvolutionExisting_b (annlLayer *layer_current, int L, int n, double *b);

// connection.c
void annlConnectFull (annlLayer *layer_previous, annlLayer *layer_current, int train);
void annlConnectFullExisting (annlLayer *layer_previous, annlLayer *layer_current, double *w, double *dw);
void annlConnectFullExisting_w (annlLayer *layer_previous, annlLayer *layer_current, double *w);
void annlConnectConvolution (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int (*a)[][2], int train);
void annlConnectConvolutionExisting_w (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int (*a)[][2], double *w);
void annlConnectPool (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int train);
void annlConnectPoolExisting_w (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, double *w);

// gradient.c
void annlCalculateGradient (annlSequence sequence, int batch_size, int b[]);
void annlCalculateGradient_omp (annlSequence sequence);
void annlCalcFull_db (annlLayer *layer_current);
void annlCalcFull_dw (annlLayer *layer_current, int layer_w_index);
void annlCalcFull_dxj (annlLayer *layer_current, int layer_w_index);
void annlCalcConvolution_db (annlLayer *layer_current);
void annlCalcConvolution_dw (annlLayer *layer_current, int layer_w_index);
void annlCalcConvolution_dxj (annlLayer *layer_current, int layer_w_index);

// integration.c
void annlUpdateParameters (annlLayer *layer_input, double step);
void annlUpdateParameters_omp (annlSequence sequence, double step);
void annlIntegrateFull_db (annlLayer *layer_current, double step);
void annlIntegrateFull_db_Adam (annlLayer *layer_current, double step);
void annlIntegrateConvolution_db (annlLayer *layer_current, double step);
void annlIntegrateConvolution_db_Adam (annlLayer *layer_current, double step);
void annlIntegrateFull_dw (annlLayer *layer_current, int layer_w_index, double step);
void annlIntegrateFull_dw_Adam (annlLayer *layer_current, int layer_w_index, double step);
void annlIntegrateConvolution_dw (annlLayer *layer_current, int layer_w_index, double step);
void annlIntegrateConvolution_dw_Adam (annlLayer *layer_current, int layer_w_index, double step);

// layer.c
annlLayer* annlCreateLayer (int size, int num_layer_w, void (*activation)(annlLayer*,int));

// loss.c
double annlCalculateLoss (int output_size, double *output, double *output_target, double *output_target_fit, int derivative, int derivative_index);
double annlCalculateLossTotal (annlSequence sequence);
double annlCalculateLossTotal_omp (annlSequence sequence);

// output.c
annlLayer* annlCalculateOutput (annlLayer *layer_input);
void annlCalcFull_z_b (annlLayer *layer_current);
void annlCalcConvolution_z_b (annlLayer *layer_current);
void annlCalcFull_z_w (annlLayer *layer_current, int layer_w_index);
void annlCalcConvolution_z_w (annlLayer *layer_current, int layer_w_index);

// randomization.c
void annlRandomizeParameters (annlLayer *layer_current, gsl_rng *rng);
void annlRandomizeParametersConvolution (annlLayer *layer_current, gsl_rng *rng);

// sequence.c
void annlLinkSequence (annlLayer *layer_previous, annlLayer *layer_next);

// training.c
void annlTrain (annlSequence sequence, annlLayer *layer_input, double loss_diff, int batch_size, gsl_rng *rng, double step, void (*status)(int,double));
void annlTrain_omp (annlSequence sequence, double loss_diff, double step, void (*status)(int,double));

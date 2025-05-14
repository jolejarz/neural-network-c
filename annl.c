// Artificial Neural Network Library for C

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <omp.h>
#include "annl.h"

annlLayer* annlCreateLayer (int size, int num_layer_w, void (*activation)(annlLayer*,int))
{
	// Create the new layer.
	annlLayer *layer_new = malloc (sizeof(annlLayer));

	layer_new->size = size;
	layer_new->layer_previous = NULL;
	layer_new->layer_next = NULL;
	layer_new->num_layer_w = num_layer_w;
	layer_new->layer_w = malloc (sizeof(annlLayer_w)*num_layer_w);
	for (int i=0; i<num_layer_w; i++) layer_new->layer_w[i].layer = NULL;
	layer_new->z = malloc (sizeof(double)*size);
	layer_new->x = malloc (sizeof(double)*size);
	layer_new->dz = malloc (sizeof(double)*size);
	layer_new->dx = malloc (sizeof(double)*size);
	layer_new->activation = activation;

	return layer_new;
}

void annlLinkSequence (annlLayer *layer_previous, annlLayer *layer_next)
{
	layer_previous->layer_next = layer_next;
	layer_next->layer_previous = layer_previous;

	return;
}

void annlSetBiasFull (annlLayer *layer_current, int train)
{
	// Set the total number of bias parameters.
	layer_current->b_num = layer_current->size;

	layer_current->b = malloc (sizeof(double)*layer_current->size);
	layer_current->db = malloc (sizeof(double)*layer_current->size);
	layer_current->update_b = 1;
	layer_current->calc_z_b = annlCalcFull_z_b;
	layer_current->calc_db = annlCalcFull_db;
	if (train==TRAIN_ADAM)
	{
		layer_current->b_m = calloc (layer_current->size, sizeof(double));
		layer_current->b_v = calloc (layer_current->size, sizeof(double));
		layer_current->integrate_db = annlIntegrateFull_db_Adam;
	}
	else
	{
		layer_current->integrate_db = annlIntegrateFull_db;
	}
	layer_current->b_index = malloc (sizeof(int)*layer_current->size);
	layer_current->b_update = malloc (sizeof(int)*layer_current->size);

	// Set the bias parameters.
	for (int j=0; j<layer_current->b_num; j++) {layer_current->b_index[j] = j; layer_current->b_update[j] = 1;}

	return;
}

void annlSetBiasFullExisting (annlLayer *layer_current, double *b, double *db)
{
	// Set the total number of bias parameters.
	layer_current->b_num = layer_current->size;

	layer_current->b = b;
	layer_current->db = db;
	layer_current->update_b = 0;
	layer_current->calc_z_b = annlCalcFull_z_b;
	layer_current->calc_db = annlCalcFull_db;
	layer_current->integrate_db = annlIntegrateFull_db;
	layer_current->b_index = malloc (sizeof(int)*layer_current->size);
	layer_current->b_update = malloc (sizeof(int)*layer_current->size);

	// Set the bias parameters.
	for (int j=0; j<layer_current->b_num; j++) {layer_current->b_index[j] = j; layer_current->b_update[j] = 1;}

	return;
}

void annlSetBiasFullExisting_b (annlLayer *layer_current, double *b)
{
	// Set the total number of bias parameters.
	layer_current->b_num = layer_current->size;

	layer_current->b = b;
	layer_current->db = malloc (sizeof(double)*layer_current->size);
	layer_current->update_b = 0;
	layer_current->calc_z_b = annlCalcFull_z_b;
	layer_current->calc_db = annlCalcFull_db;
	layer_current->integrate_db = annlIntegrateFull_db;
	layer_current->b_index = malloc (sizeof(int)*layer_current->size);
	layer_current->b_update = malloc (sizeof(int)*layer_current->size);

	// Set the bias parameters.
	for (int j=0; j<layer_current->b_num; j++) {layer_current->b_index[j] = j; layer_current->b_update[j] = 1;}

	return;
}

void annlSetBiasLeNet (annlLayer *layer_current, int L, int n, int train)
{
	// Set the total number of bias parameters.
	layer_current->b_num = n;

	layer_current->b_value = malloc (sizeof(double)*layer_current->size);
	layer_current->db = malloc (sizeof(double)*layer_current->size);
	layer_current->update_b = 1;
	layer_current->calc_z_b = annlCalcLeNet_z_b;
	layer_current->calc_db = annlCalcLeNet_db;
	if (train==TRAIN_ADAM)
	{
		layer_current->b_m = calloc (layer_current->size, sizeof(double));
		layer_current->b_v = calloc (layer_current->size, sizeof(double));
		layer_current->integrate_db = annlIntegrateLeNet_db_Adam;
	}
	else
	{
		layer_current->integrate_db = annlIntegrateLeNet_db;
	}
	layer_current->b_parameters_trainable = malloc (sizeof(double)*layer_current->size);
	layer_current->bi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_xi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_list_trainable = malloc (sizeof(annlIndex_b)*layer_current->size);
	layer_current->b_parameters_not_trainable = malloc (sizeof(double)*layer_current->size);
	layer_current->bi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_xi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_list_not_trainable = malloc (sizeof(annlIndex_b)*layer_current->size);

	int b_list_index=0;

	int b_parameter_index;
	int xi_num;

	int *b_list_index_start = calloc (n, sizeof(int));
	int *b_list_index_previous = calloc (n, sizeof(int));

	int *b_xi_list_index_start = calloc (layer_current->size, sizeof(int));
	int *b_xi_list_index_previous = calloc (layer_current->size, sizeof(int));

	layer_current->b_xi_num=0;

	for (int k=0; k<n; k++)
	{
		b_parameter_index = k;

		for (int i=0; i<L; i++)
		{
			for (int j=0; j<L; j++)
			{
				xi_num = L*L*k+L*i+j;

				layer_current->bi_list_trainable[b_list_index].bi_index = b_parameter_index;
				layer_current->bi_list_trainable[b_list_index].xi_index = xi_num;

				if (b_list_index_start[b_parameter_index]==1)
				{
					layer_current->bi_list_trainable[b_list_index].bi_list_previous = b_list_index_previous[b_parameter_index];
					layer_current->bi_list_trainable[b_list_index].bi_list_next = b_list_index;
					layer_current->bi_list_trainable[b_list_index_previous[b_parameter_index]].bi_list_next = b_list_index;
				}
				else
				{
					layer_current->bi_list_trainable[b_list_index].bi_list_previous = b_list_index;
					layer_current->bi_list_trainable[b_list_index].bi_list_next = b_list_index;
					layer_current->bi_list_start_trainable[b_parameter_index]=b_list_index;
					b_list_index_start[b_parameter_index]=1;
				}
				if (b_xi_list_index_start[xi_num]==1)
				{
					layer_current->bi_list_trainable[b_list_index].xi_list_previous = b_xi_list_index_previous[xi_num];
					layer_current->bi_list_trainable[b_list_index].xi_list_next = b_list_index;
					layer_current->bi_list_trainable[b_xi_list_index_previous[xi_num]].xi_list_next = b_list_index;
				}
				else
				{
					layer_current->b_xi_num++;
					layer_current->bi_list_trainable[b_list_index].xi_list_previous = b_list_index;
					layer_current->bi_list_trainable[b_list_index].xi_list_next = b_list_index;
					layer_current->bi_xi_list_start_trainable[xi_num]=b_list_index;
					b_xi_list_index_start[xi_num]=1;
				}

				b_list_index_previous[b_parameter_index] = b_list_index;
				b_xi_list_index_previous[xi_num] = b_list_index;

				b_list_index++;
			}
		}
	}

	free (b_xi_list_index_previous);
	free (b_xi_list_index_start);

	free (b_list_index_previous);
	free (b_list_index_start);

	return;
}

void annlSetBiasLeNetExisting_b (annlLayer *layer_current, int L, int n, double *b)
{
	// Set the total number of bias parameters.
	layer_current->b_num = n;

	layer_current->b_value = b;
	layer_current->db = malloc (sizeof(double)*layer_current->size);
	layer_current->update_b = 0;
	layer_current->calc_z_b = annlCalcLeNet_z_b;
	layer_current->calc_db = annlCalcLeNet_db;
	layer_current->integrate_db = annlIntegrateLeNet_db;
	layer_current->b_parameters_trainable = malloc (sizeof(double)*layer_current->size);
	layer_current->bi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_xi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_list_trainable = malloc (sizeof(annlIndex_b)*layer_current->size);
	layer_current->b_parameters_not_trainable = malloc (sizeof(double)*layer_current->size);
	layer_current->bi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_xi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->bi_list_not_trainable = malloc (sizeof(annlIndex_b)*layer_current->size);

	int b_list_index=0;

	int b_parameter_index;
	int xi_num;

	int *b_list_index_start = calloc (n, sizeof(int));
	int *b_list_index_previous = calloc (n, sizeof(int));

	int *b_xi_list_index_start = calloc (layer_current->size, sizeof(int));
	int *b_xi_list_index_previous = calloc (layer_current->size, sizeof(int));

	layer_current->b_xi_num=0;

	for (int k=0; k<n; k++)
	{
		b_parameter_index = k;

		for (int i=0; i<L; i++)
		{
			for (int j=0; j<L; j++)
			{
				xi_num = L*L*k+L*i+j;

				layer_current->bi_list_trainable[b_list_index].bi_index = b_parameter_index;
				layer_current->bi_list_trainable[b_list_index].xi_index = xi_num;

				if (b_list_index_start[b_parameter_index]==1)
				{
					layer_current->bi_list_trainable[b_list_index].bi_list_previous = b_list_index_previous[b_parameter_index];
					layer_current->bi_list_trainable[b_list_index].bi_list_next = b_list_index;
					layer_current->bi_list_trainable[b_list_index_previous[b_parameter_index]].bi_list_next = b_list_index;
				}
				else
				{
					layer_current->bi_list_trainable[b_list_index].bi_list_previous = b_list_index;
					layer_current->bi_list_trainable[b_list_index].bi_list_next = b_list_index;
					layer_current->bi_list_start_trainable[b_parameter_index]=b_list_index;
					b_list_index_start[b_parameter_index]=1;
				}
				if (b_xi_list_index_start[xi_num]==1)
				{
					layer_current->bi_list_trainable[b_list_index].xi_list_previous = b_xi_list_index_previous[xi_num];
					layer_current->bi_list_trainable[b_list_index].xi_list_next = b_list_index;
					layer_current->bi_list_trainable[b_xi_list_index_previous[xi_num]].xi_list_next = b_list_index;
				}
				else
				{
					layer_current->b_xi_num++;
					layer_current->bi_list_trainable[b_list_index].xi_list_previous = b_list_index;
					layer_current->bi_list_trainable[b_list_index].xi_list_next = b_list_index;
					layer_current->bi_xi_list_start_trainable[xi_num]=b_list_index;
					b_xi_list_index_start[xi_num]=1;
				}

				b_list_index_previous[b_parameter_index] = b_list_index;
				b_xi_list_index_previous[xi_num] = b_list_index;

				b_list_index++;
			}
		}
	}

	free (b_xi_list_index_previous);
	free (b_xi_list_index_start);

	free (b_list_index_previous);
	free (b_list_index_start);

	return;
}

void annlCalcFull_z_b (annlLayer *layer_current)
{
	int i_max = layer_current->size;

	for (int i=0; i<i_max; i++)
	{
		layer_current->z[i] += layer_current->b[i];
	}

	return;
}

void annlCalcFull_db (annlLayer *layer_current)
{
	int i_max = layer_current->size;

	for (int i=0; i<i_max; i++)
	{
		layer_current->db[i] += layer_current->dx[i] * layer_current->dz[i];
	}

	return;
}

void annlIntegrateFull_db (annlLayer *layer_current, double step)
{
	int i_max = layer_current->size;

	for (int i=0; i<i_max; i++)
	{
		layer_current->b[i] -= layer_current->db[i]*step;
	}

	return;
}

void annlIntegrateFull_db_Adam (annlLayer *layer_current, double step)
{
	const double beta1 = 0.9, beta2 = 0.999, epsilon = 0.00000001;

	int i_max = layer_current->size;

	for (int i=0; i<i_max; i++)
	{
		layer_current->b_m[i] = beta1*layer_current->b_m[i] + (1-beta1)*layer_current->db[i];
		layer_current->b_v[i] = beta2*layer_current->b_v[i] + (1-beta2)*layer_current->db[i]*layer_current->db[i];
		layer_current->b[i] -= layer_current->b_m[i]/(1-beta1)/(sqrt(layer_current->b_v[i]/(1-beta2))+epsilon)*step;
	}

	return;
}

void annlCalcLeNet_z_b (annlLayer *layer_current)
{
	for (int k=0; k<layer_current->b_xi_num; k++)
	{
		int index_previous;
		int index_current = layer_current->bi_xi_list_start_trainable[k];

		do
		{
			int bi_index = layer_current->bi_list_trainable[index_current].bi_index;
			int xi_index = layer_current->bi_list_trainable[index_current].xi_index;

			layer_current->z[xi_index] += layer_current->b_value[bi_index];

			index_previous = index_current;
			index_current = layer_current->bi_list_trainable[index_current].xi_list_next;
		}
		while (index_current!=index_previous);
	}

	return;
}

void annlCalcLeNet_db (annlLayer *layer_current)
{
	for (int k=0; k<layer_current->b_num; k++)
	{
		int index_previous;
		int index_current = layer_current->bi_list_start_trainable[k];

		do
		{
			int bi_index = layer_current->bi_list_trainable[index_current].bi_index;
			int xi_index = layer_current->bi_list_trainable[index_current].xi_index;

			layer_current->db[bi_index] += layer_current->dx[xi_index] * layer_current->dz[xi_index];

			index_previous = index_current;
			index_current = layer_current->bi_list_trainable[index_current].bi_list_next;
		}
		while (index_current!=index_previous);
	}

	return;
}

void annlIntegrateLeNet_db (annlLayer *layer_current, double step)
{
	for (int k=0; k<layer_current->b_num; k++)
	{
		layer_current->b_value[k] -= step * layer_current->db[k];
	}

	return;
}

void annlIntegrateLeNet_db_Adam (annlLayer *layer_current, double step)
{
	const double beta1 = 0.9, beta2 = 0.999, epsilon = 0.00000001;

	for (int k=0; k<layer_current->b_num; k++)
	{
		layer_current->b_m[k] = beta1*layer_current->b_m[k] + (1-beta1)*layer_current->db[k];
		layer_current->b_v[k] = beta2*layer_current->b_v[k] + (1-beta2)*layer_current->db[k]*layer_current->db[k];
		layer_current->b_value[k] -= layer_current->b_m[k]/(1-beta1)/(sqrt(layer_current->b_v[k]/(1-beta2))+epsilon)*step;
	}

	return;
}

void annlConnectFull (annlLayer *layer_previous, annlLayer *layer_current, int train)
{
	signed int i=-1;
	
	while (layer_current->layer_w[++i].layer!=NULL);

	layer_current->layer_w[i].layer = layer_previous;
	layer_current->layer_w[i].w = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[i].dw = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[i].update_w = 1;
	layer_current->layer_w[i].calc_z_w = annlCalcFull_z_w;
	layer_current->layer_w[i].calc_dw = annlCalcFull_dw;
	layer_current->layer_w[i].calc_dxj = annlCalcFull_dxj;
	if (train==TRAIN_ADAM)
	{
		layer_current->layer_w[i].w_m = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[i].w_v = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[i].integrate_dw = annlIntegrateFull_dw_Adam;
	}
	else
	{
		layer_current->layer_w[i].integrate_dw = annlIntegrateFull_dw;
	}
	layer_current->layer_w[i].w_index = malloc (sizeof(int)*layer_current->size*layer_previous->size);
	layer_current->layer_w[i].w_update = malloc (sizeof(int)*layer_current->size*layer_previous->size);

	// Set the total number of weight parameters.
	layer_current->layer_w[i].w_num = layer_current->size * layer_previous->size;

	// Connect the weight parameters.
	for (int j=0; j<layer_current->layer_w[i].w_num; j++) {layer_current->layer_w[i].w_index[j] = j; layer_current->layer_w[i].w_update[j] = 1;}

	return;
}

void annlConnectFullExisting (annlLayer *layer_previous, annlLayer *layer_current, double *w, double *dw)
{
	signed int i=-1;
	
	while (layer_current->layer_w[++i].layer!=NULL);

	layer_current->layer_w[i].layer = layer_previous;
	layer_current->layer_w[i].w = w;
	layer_current->layer_w[i].dw = dw;
	layer_current->layer_w[i].update_w = 0;
	layer_current->layer_w[i].calc_z_w = annlCalcFull_z_w;
	layer_current->layer_w[i].calc_dw = annlCalcFull_dw;
	layer_current->layer_w[i].calc_dxj = annlCalcFull_dxj;
	layer_current->layer_w[i].integrate_dw = annlIntegrateFull_dw;
	layer_current->layer_w[i].w_index = malloc (sizeof(int)*layer_current->size*layer_previous->size);
	layer_current->layer_w[i].w_update = malloc (sizeof(int)*layer_current->size*layer_previous->size);

	// Set the total number of weight parameters.
	layer_current->layer_w[i].w_num = layer_current->size * layer_previous->size;

	// Connect the weight parameters.
	for (int j=0; j<layer_current->layer_w[i].w_num; j++) {layer_current->layer_w[i].w_index[j] = j; layer_current->layer_w[i].w_update[j] = 1;}

	return;
}

void annlConnectFullExisting_w (annlLayer *layer_previous, annlLayer *layer_current, double *w)
{
	signed int i=-1;
	
	while (layer_current->layer_w[++i].layer!=NULL);

	layer_current->layer_w[i].layer = layer_previous;
	layer_current->layer_w[i].w = w;
	layer_current->layer_w[i].dw = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[i].update_w = 0;
	layer_current->layer_w[i].calc_z_w = annlCalcFull_z_w;
	layer_current->layer_w[i].calc_dw = annlCalcFull_dw;
	layer_current->layer_w[i].calc_dxj = annlCalcFull_dxj;
	layer_current->layer_w[i].integrate_dw = annlIntegrateFull_dw;
	layer_current->layer_w[i].w_index = malloc (sizeof(int)*layer_current->size*layer_previous->size);
	layer_current->layer_w[i].w_update = malloc (sizeof(int)*layer_current->size*layer_previous->size);

	// Set the total number of weight parameters.
	layer_current->layer_w[i].w_num = layer_current->size * layer_previous->size;

	// Connect the weight parameters.
	for (int j=0; j<layer_current->layer_w[i].w_num; j++) {layer_current->layer_w[i].w_index[j] = j; layer_current->layer_w[i].w_update[j] = 1;}

	return;
}

void annlConnectConvolution (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int (*a)[][2], int train)
{
	signed int layer_w_index=-1;
	
	while (layer_current->layer_w[++layer_w_index].layer!=NULL);

	layer_current->layer_w[layer_w_index].layer = layer_previous;
	layer_current->layer_w[layer_w_index].w_value = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[layer_w_index].dw = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[layer_w_index].update_w = 1;
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcLeNet_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcLeNet_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcLeNet_dxj;
	if (train==TRAIN_ADAM)
	{
		layer_current->layer_w[layer_w_index].w_m = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].w_v = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateLeNet_dw_Adam;
	}
	else
	{
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateLeNet_dw;
	}
	layer_current->layer_w[layer_w_index].w_parameters_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].w_parameters_not_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_not_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_not_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_not_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));

	// Set the total number of weight parameters.
	layer_current->layer_w[layer_w_index].w_num = n*5*5;

	int w_list_index=0;

	int w_parameter_index;
	int xj_num;
	int xi_num;

	int *w_list_index_start = calloc (n*5*5, sizeof(int));
	int *w_list_index_previous = calloc (n*5*5, sizeof(int));

	int *xj_list_index_start = calloc (layer_previous->size, sizeof(int));
	int *xj_list_index_previous = calloc (layer_previous->size, sizeof(int));

	int *xi_list_index_start = calloc (layer_current->size, sizeof(int));
	int *xi_list_index_previous = calloc (layer_current->size, sizeof(int));

	layer_current->layer_w[layer_w_index].w_xj_num=0;
	layer_current->layer_w[layer_w_index].w_xi_num=0;

	for (int k=0; k<n; k++)
	{
		for (int i=0; i<L-4; i++)
		{
			for (int j=0; j<L-4; j++)
			{
				int i_center = 2+i;
				int j_center = 2+j;

				xi_num = (L-4)*(L-4)*(*a)[k][1]+(L-4)*i+j;

				for (int m=-2; m<=2; m++)
				{
					for (int n=-2; n<=2; n++)
					{
						w_parameter_index = 5*5*k+5*(2+m)+(2+n);
						xj_num = L*L*(*a)[k][0]+L*(i_center+m)+(j_center+n);

						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_index = w_parameter_index;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_index = xj_num;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_index = xi_num;

						if (w_list_index_start[w_parameter_index]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index_previous[w_parameter_index];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index_previous[w_parameter_index]].wij_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_start_trainable[w_parameter_index]=w_list_index;
							w_list_index_start[w_parameter_index]=1;
						}
						if (xj_list_index_start[xj_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = xj_list_index_previous[xj_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xj_list_index_previous[xj_num]].xj_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xj_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable[xj_num]=w_list_index;
							xj_list_index_start[xj_num]=1;
						}
						if (xi_list_index_start[xi_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = xi_list_index_previous[xi_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xi_list_index_previous[xi_num]].xi_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xi_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable[xi_num]=w_list_index;
							xi_list_index_start[xi_num]=1;
						}

						w_list_index_previous[w_parameter_index] = w_list_index;
						xj_list_index_previous[xj_num] = w_list_index;
						xi_list_index_previous[xi_num] = w_list_index;

						w_list_index++;
					}
				}
			}
		}
	}

	free (xi_list_index_previous);
	free (xi_list_index_start);

	free (xj_list_index_previous);
	free (xj_list_index_start);

	free (w_list_index_previous);
	free (w_list_index_start);

	return;
}

void annlConnectConvolutionExisting_w (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int (*a)[][2], double *w)
{
	signed int layer_w_index=-1;
	
	while (layer_current->layer_w[++layer_w_index].layer!=NULL);

	layer_current->layer_w[layer_w_index].layer = layer_previous;
	layer_current->layer_w[layer_w_index].w_value = w;
	layer_current->layer_w[layer_w_index].dw = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[layer_w_index].update_w = 0;
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcLeNet_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcLeNet_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcLeNet_dxj;
	layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateLeNet_dw;
	layer_current->layer_w[layer_w_index].w_parameters_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].w_parameters_not_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_not_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_not_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_not_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));

	// Set the total number of weight parameters.
	layer_current->layer_w[layer_w_index].w_num = n*5*5;

	int w_list_index=0;

	int w_parameter_index;
	int xj_num;
	int xi_num;

	int *w_list_index_start = calloc (n*5*5, sizeof(int));
	int *w_list_index_previous = calloc (n*5*5, sizeof(int));

	int *xj_list_index_start = calloc (layer_previous->size, sizeof(int));
	int *xj_list_index_previous = calloc (layer_previous->size, sizeof(int));

	int *xi_list_index_start = calloc (layer_current->size, sizeof(int));
	int *xi_list_index_previous = calloc (layer_current->size, sizeof(int));

	layer_current->layer_w[layer_w_index].w_xj_num=0;
	layer_current->layer_w[layer_w_index].w_xi_num=0;

	for (int k=0; k<n; k++)
	{
		for (int i=0; i<L-4; i++)
		{
			for (int j=0; j<L-4; j++)
			{
				int i_center = 2+i;
				int j_center = 2+j;

				xi_num = (L-4)*(L-4)*(*a)[k][1]+(L-4)*i+j;

				for (int m=-2; m<=2; m++)
				{
					for (int n=-2; n<=2; n++)
					{
						w_parameter_index = 5*5*k+5*(2+m)+(2+n);
						xj_num = L*L*(*a)[k][0]+L*(i_center+m)+(j_center+n);

						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_index = w_parameter_index;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_index = xj_num;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_index = xi_num;

						if (w_list_index_start[w_parameter_index]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index_previous[w_parameter_index];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index_previous[w_parameter_index]].wij_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_start_trainable[w_parameter_index]=w_list_index;
							w_list_index_start[w_parameter_index]=1;
						}
						if (xj_list_index_start[xj_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = xj_list_index_previous[xj_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xj_list_index_previous[xj_num]].xj_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xj_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable[xj_num]=w_list_index;
							xj_list_index_start[xj_num]=1;
						}
						if (xi_list_index_start[xi_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = xi_list_index_previous[xi_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xi_list_index_previous[xi_num]].xi_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xi_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable[xi_num]=w_list_index;
							xi_list_index_start[xi_num]=1;
						}

						w_list_index_previous[w_parameter_index] = w_list_index;
						xj_list_index_previous[xj_num] = w_list_index;
						xi_list_index_previous[xi_num] = w_list_index;

						w_list_index++;
					}
				}
			}
		}
	}

	free (xi_list_index_previous);
	free (xi_list_index_start);

	free (xj_list_index_previous);
	free (xj_list_index_start);

	free (w_list_index_previous);
	free (w_list_index_start);

	return;
}

void annlConnectPool (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, int train)
{
	signed int layer_w_index=-1;
	
	while (layer_current->layer_w[++layer_w_index].layer!=NULL);

	layer_current->layer_w[layer_w_index].layer = layer_previous;
	layer_current->layer_w[layer_w_index].w_value = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[layer_w_index].dw = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[layer_w_index].update_w = 1;
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcLeNet_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcLeNet_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcLeNet_dxj;
	if (train==TRAIN_ADAM)
	{
		layer_current->layer_w[layer_w_index].w_m = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].w_v = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateLeNet_dw_Adam;
	}
	else
	{
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateLeNet_dw;
	}
	layer_current->layer_w[layer_w_index].w_parameters_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].w_parameters_not_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_not_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_not_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_not_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));

	// Set the total number of weight parameters.
	layer_current->layer_w[layer_w_index].w_num = n;

	int w_list_index=0;

	int w_parameter_index;
	int xj_num;
	int xi_num;

	int *w_list_index_start = calloc (n, sizeof(int));
	int *w_list_index_previous = calloc (n, sizeof(int));

	int *xj_list_index_start = calloc (layer_previous->size, sizeof(int));
	int *xj_list_index_previous = calloc (layer_previous->size, sizeof(int));

	int *xi_list_index_start = calloc (layer_current->size, sizeof(int));
	int *xi_list_index_previous = calloc (layer_current->size, sizeof(int));

	layer_current->layer_w[layer_w_index].w_xj_num=0;
	layer_current->layer_w[layer_w_index].w_xi_num=0;

	for (int k=0; k<n; k++)
	{
		for (int i=0; i<L/2; i++)
		{
			for (int j=0; j<L/2; j++)
			{
				xi_num = (L/2)*(L/2)*k+(L/2)*i+j;

				for (int m=0; m<2; m++)
				{
					for (int n=0; n<2; n++)
					{
						w_parameter_index = k;
						xj_num = L*L*k+L*(2*i+m)+(2*j+n);

						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_index = w_parameter_index;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_index = xj_num;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_index = xi_num;

						if (w_list_index_start[w_parameter_index]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index_previous[w_parameter_index];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index_previous[w_parameter_index]].wij_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_start_trainable[w_parameter_index]=w_list_index;
							w_list_index_start[w_parameter_index]=1;
						}
						if (xj_list_index_start[xj_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = xj_list_index_previous[xj_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xj_list_index_previous[xj_num]].xj_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xj_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable[xj_num]=w_list_index;
							xj_list_index_start[xj_num]=1;
						}
						if (xi_list_index_start[xi_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = xi_list_index_previous[xi_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xi_list_index_previous[xi_num]].xi_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xi_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable[xi_num]=w_list_index;
							xi_list_index_start[xi_num]=1;
						}

						w_list_index_previous[w_parameter_index] = w_list_index;
						xj_list_index_previous[xj_num] = w_list_index;
						xi_list_index_previous[xi_num] = w_list_index;

						w_list_index++;
					}
				}
			}
		}
	}

	free (xi_list_index_previous);
	free (xi_list_index_start);

	free (xj_list_index_previous);
	free (xj_list_index_start);

	free (w_list_index_previous);
	free (w_list_index_start);

	return;
}

void annlConnectPoolExisting_w (annlLayer *layer_previous, annlLayer *layer_current, int L, int n, double *w)
{
	signed int layer_w_index=-1;
	
	while (layer_current->layer_w[++layer_w_index].layer!=NULL);

	layer_current->layer_w[layer_w_index].layer = layer_previous;
	layer_current->layer_w[layer_w_index].w_value = w;
	layer_current->layer_w[layer_w_index].dw = malloc (sizeof(double)*layer_current->size*layer_previous->size);
	layer_current->layer_w[layer_w_index].update_w = 0;
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcLeNet_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcLeNet_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcLeNet_dxj;
	layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateLeNet_dw;
	layer_current->layer_w[layer_w_index].w_parameters_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].w_parameters_not_trainable = malloc (sizeof(double)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_list_start_not_trainable = malloc (sizeof(int)*layer_current->size*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xj_list_start_not_trainable = malloc (sizeof(int)*(layer_previous->size));
	layer_current->layer_w[layer_w_index].wij_xi_list_start_not_trainable = malloc (sizeof(int)*layer_current->size);
	layer_current->layer_w[layer_w_index].wij_list_not_trainable = malloc (sizeof(annlIndex_w)*layer_current->size*(layer_previous->size));

	// Set the total number of weight parameters.
	layer_current->layer_w[layer_w_index].w_num = n;

	int w_list_index=0;

	int w_parameter_index;
	int xj_num;
	int xi_num;

	int *w_list_index_start = calloc (n, sizeof(int));
	int *w_list_index_previous = calloc (n, sizeof(int));

	int *xj_list_index_start = calloc (layer_previous->size, sizeof(int));
	int *xj_list_index_previous = calloc (layer_previous->size, sizeof(int));

	int *xi_list_index_start = calloc (layer_current->size, sizeof(int));
	int *xi_list_index_previous = calloc (layer_current->size, sizeof(int));

	layer_current->layer_w[layer_w_index].w_xj_num=0;
	layer_current->layer_w[layer_w_index].w_xi_num=0;

	for (int k=0; k<n; k++)
	{
		for (int i=0; i<L/2; i++)
		{
			for (int j=0; j<L/2; j++)
			{
				xi_num = (L/2)*(L/2)*k+(L/2)*i+j;

				for (int m=0; m<2; m++)
				{
					for (int n=0; n<2; n++)
					{
						w_parameter_index = k;
						xj_num = L*L*k+L*(2*i+m)+(2*j+n);

						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_index = w_parameter_index;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_index = xj_num;
						layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_index = xi_num;

						if (w_list_index_start[w_parameter_index]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index_previous[w_parameter_index];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index_previous[w_parameter_index]].wij_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].wij_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_start_trainable[w_parameter_index]=w_list_index;
							w_list_index_start[w_parameter_index]=1;
						}
						if (xj_list_index_start[xj_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = xj_list_index_previous[xj_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xj_list_index_previous[xj_num]].xj_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xj_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xj_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable[xj_num]=w_list_index;
							xj_list_index_start[xj_num]=1;
						}
						if (xi_list_index_start[xi_num]==1)
						{
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = xi_list_index_previous[xi_num];
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[xi_list_index_previous[xi_num]].xi_list_next = w_list_index;
						}
						else
						{
							layer_current->layer_w[layer_w_index].w_xi_num++;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_previous = w_list_index;
							layer_current->layer_w[layer_w_index].wij_list_trainable[w_list_index].xi_list_next = w_list_index;
							layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable[xi_num]=w_list_index;
							xi_list_index_start[xi_num]=1;
						}

						w_list_index_previous[w_parameter_index] = w_list_index;
						xj_list_index_previous[xj_num] = w_list_index;
						xi_list_index_previous[xi_num] = w_list_index;

						w_list_index++;
					}
				}
			}
		}
	}

	free (xi_list_index_previous);
	free (xi_list_index_start);

	free (xj_list_index_previous);
	free (xj_list_index_start);

	free (w_list_index_previous);
	free (w_list_index_start);

	return;
}

void annlCalcFull_z_w (annlLayer *layer_current, int layer_w_index)
{
	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	int i_max = layer_current->size;
	int j_max = layer_previous->size;

	for (int i=0; i<i_max; i++)
	{
		for (int j=0; j<j_max; j++)
		{
			layer_current->z[i] += layer_current->layer_w[layer_w_index].w[j_max*i+j] * layer_previous->x[j];
		}
	}

	return;
}

void annlCalcFull_dw (annlLayer *layer_current, int layer_w_index)
{
	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	int i_max = layer_current->size;
	int j_max = layer_previous->size;

	for (int i=0; i<i_max; i++)
	{
		for (int j=0; j<j_max; j++)
		{
			layer_current->layer_w[layer_w_index].dw[j_max*i+j] += layer_current->dx[i] * layer_current->dz[i] * layer_previous->x[j];
		}
	}

	return;
}

void annlCalcFull_dxj (annlLayer *layer_current, int layer_w_index)
{
	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	int i_max = layer_current->size;
	int j_max = layer_previous->size;

	for (int j=0; j<j_max; j++)
	{
		layer_previous->dx[j] = 0;

		for (int i=0; i<i_max; i++)
		{
			layer_previous->dx[j] += layer_current->dx[i] * layer_current->dz[i] * layer_current->layer_w[layer_w_index].w[j_max*i+j];
		}
	}

	return;
}

void annlIntegrateFull_dw (annlLayer *layer_current, int layer_w_index, double step)
{
	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	int i_max = layer_current->size;
	int j_max = layer_previous->size;

	for (int i=0; i<i_max; i++)
	{
		for (int j=0; j<j_max; j++)
		{
			layer_current->layer_w[layer_w_index].w[j_max*i+j] -= layer_current->layer_w[layer_w_index].dw[j_max*i+j]*step;
		}
	}

	return;
}

void annlIntegrateFull_dw_Adam (annlLayer *layer_current, int layer_w_index, double step)
{
	const double beta1 = 0.9, beta2 = 0.999, epsilon = 0.00000001;

	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	int i_max = layer_current->size;
	int j_max = layer_previous->size;

	for (int i=0; i<i_max; i++)
	{
		for (int j=0; j<j_max; j++)
		{
			layer_current->layer_w[layer_w_index].w_m[j_max*i+j] = beta1*layer_current->layer_w[layer_w_index].w_m[j_max*i+j] + (1-beta1)*layer_current->layer_w[layer_w_index].dw[j_max*i+j];
			layer_current->layer_w[layer_w_index].w_v[j_max*i+j] = beta2*layer_current->layer_w[layer_w_index].w_v[j_max*i+j] + (1-beta2)*layer_current->layer_w[layer_w_index].dw[j_max*i+j]*layer_current->layer_w[layer_w_index].dw[j_max*i+j];
			layer_current->layer_w[layer_w_index].w[j_max*i+j] -= layer_current->layer_w[layer_w_index].w_m[j_max*i+j]/(1-beta1)/(sqrt(layer_current->layer_w[layer_w_index].w_v[j_max*i+j]/(1-beta2))+epsilon)*step;
		}
	}

	return;
}

void annlCalcLeNet_z_w (annlLayer *layer_current, int layer_w_index)
{
	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	for (int k=0; k<layer_current->layer_w[layer_w_index].w_xi_num; k++)
	{
		int index_previous;
		int index_current = layer_current->layer_w[layer_w_index].wij_xi_list_start_trainable[k];

		do
		{
			int w_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].wij_index;
			int xj_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xj_index;
			int xi_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xi_index;

			layer_current->z[xi_index] += layer_current->layer_w[layer_w_index].w_value[w_index] * layer_previous->x[xj_index];

			index_previous = index_current;
			index_current = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xi_list_next;
		}
		while (index_current!=index_previous);
	}

	return;
}

void annlCalcLeNet_dw (annlLayer *layer_current, int layer_w_index)
{
	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	for (int k=0; k<layer_current->layer_w[layer_w_index].w_num; k++)
	{
		int index_previous;
		int index_current = layer_current->layer_w[layer_w_index].wij_list_start_trainable[k];

		do
		{
			int w_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].wij_index;
			int xj_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xj_index;
			int xi_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xi_index;

			layer_current->layer_w[layer_w_index].dw[w_index] += layer_current->dx[xi_index] * layer_current->dz[xi_index] * layer_previous->x[xj_index];

			index_previous = index_current;
			index_current = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].wij_list_next;
		}
		while (index_current!=index_previous);
	}

	return;
}

void annlCalcLeNet_dxj (annlLayer *layer_current, int layer_w_index)
{
	annlLayer *layer_previous = layer_current->layer_w[layer_w_index].layer;

	for (int k=0; k<layer_current->layer_w[layer_w_index].w_xj_num; k++)
	{
		int index_previous;
		int index_current = layer_current->layer_w[layer_w_index].wij_xj_list_start_trainable[k];

		int xj_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xj_index;

		layer_previous->dx[xj_index] = 0;

		do
		{
			int w_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].wij_index;
			int xj_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xj_index;
			int xi_index = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xi_index;

			layer_previous->dx[xj_index] += layer_current->dx[xi_index] * layer_current->dz[xi_index] * layer_current->layer_w[layer_w_index].w_value[w_index];

			index_previous = index_current;
			index_current = layer_current->layer_w[layer_w_index].wij_list_trainable[index_current].xj_list_next;
		}
		while (index_current!=index_previous);
	}

	return;
}

void annlIntegrateLeNet_dw (annlLayer *layer_current, int layer_w_index, double step)
{
	for (int k=0; k<layer_current->layer_w[layer_w_index].w_num; k++)
	{
		layer_current->layer_w[layer_w_index].w_value[k] -= step * layer_current->layer_w[layer_w_index].dw[k];
	}

	return;
}

void annlIntegrateLeNet_dw_Adam (annlLayer *layer_current, int layer_w_index, double step)
{
	const double beta1 = 0.9, beta2 = 0.999, epsilon = 0.00000001;

	for (int k=0; k<layer_current->layer_w[layer_w_index].w_num; k++)
	{
		layer_current->layer_w[layer_w_index].w_m[k] = beta1*layer_current->layer_w[layer_w_index].w_m[k] + (1-beta1)*layer_current->layer_w[layer_w_index].dw[k];
		layer_current->layer_w[layer_w_index].w_v[k] = beta2*layer_current->layer_w[layer_w_index].w_v[k] + (1-beta2)*layer_current->layer_w[layer_w_index].dw[k]*layer_current->layer_w[layer_w_index].dw[k];
		layer_current->layer_w[layer_w_index].w_value[k] -= layer_current->layer_w[layer_w_index].w_m[k]/(1-beta1)/(sqrt(layer_current->layer_w[layer_w_index].w_v[k]/(1-beta2))+epsilon)*step;
	}

	return;
}

void annlRandomizeParameters (annlLayer *layer_current, gsl_rng *rng)
{
	// Randomize the weight parameters.
	for (int layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++)
	{
		for (int i=0; i<layer_current->layer_w[layer_w_index].w_num; i++)
		{
			layer_current->layer_w[layer_w_index].dw[i] = 2*(gsl_rng_uniform(rng)-0.5);
		}
	}

	// Randomize the bias parameters.
	for (int i=0; i<layer_current->b_num; i++) {layer_current->db[i] = 2*(gsl_rng_uniform(rng)-0.5);}

	// Update the weight parameters.
	for (int layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++)
	{
		for (int i=0; i<layer_current->layer_w[layer_w_index].w_num; i++)
		{
			if (layer_current->layer_w[layer_w_index].w_update[i]==1) layer_current->layer_w[layer_w_index].w[i] = layer_current->layer_w[layer_w_index].dw[layer_current->layer_w[layer_w_index].w_index[i]];
		}
	}

	// Update the bias parameters.
	for (int i=0; i<layer_current->size; i++)
	{
		if (layer_current->b_update[i]==1) layer_current->b[i] = layer_current->db[layer_current->b_index[i]];
	}

	return;
}

void annlRandomizeParametersLeNet (annlLayer *layer_current, gsl_rng *rng)
{
	// Randomize the weight parameters.
	for (int layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++)
	{
		for (int i=0; i<layer_current->layer_w[layer_w_index].w_num; i++)
		{
			layer_current->layer_w[layer_w_index].dw[i] = 2*(gsl_rng_uniform(rng)-0.5);
		}
	}

	// Randomize the bias parameters.
	for (int i=0; i<layer_current->b_num; i++) {layer_current->db[i] = 2*(gsl_rng_uniform(rng)-0.5);}

	// Update the weight parameters.
	for (int layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++)
	{
		for (int i=0; i<layer_current->layer_w[layer_w_index].w_num; i++)
		{
			layer_current->layer_w[layer_w_index].w_value[i] = layer_current->layer_w[layer_w_index].dw[i];
		}
	}

	// Update the bias parameters.
	for (int i=0; i<layer_current->size; i++)
	{
		layer_current->b_value[i] = layer_current->db[i];
	}

	return;
}

annlLayer* annlCalculateOutput (annlLayer *layer_input)
{
	int i, j, i_max, j_max, layer_w_index;

	annlLayer *layer_previous = layer_input;

	annlLayer *layer_current = layer_input->layer_next;

	while (layer_current!=NULL)
	{
		i_max = layer_current->size;

		for (i=0; i<i_max; i++) layer_current->z[i] = 0;

		layer_previous = layer_current;
		layer_current = layer_current->layer_next;
	}

	layer_current = layer_input->layer_next;

	while (layer_current!=NULL)
	{
		for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++) (*(layer_current->layer_w[layer_w_index].calc_z_w))(layer_current,layer_w_index);

		(*(layer_current->calc_z_b))(layer_current);

		(*(layer_current->activation))(layer_current,NO_DERIVATIVE);

		layer_previous = layer_current;
		layer_current = layer_current->layer_next;
	}

	return layer_previous;
}

double annlCalculateLoss (int output_size, double *output, double *output_target, double *output_target_fit, int derivative, int derivative_index)
{
	double x=0;

	if (derivative==NO_DERIVATIVE)
	{
		for (int i=0; i<output_size; i++)
		{
			if (*(output_target_fit+i)==1) x += ((*(output+i))-(*(output_target+i))) * ((*(output+i))-(*(output_target+i)));
		}
	}
	else if (*(output_target_fit+derivative_index)==1) x = 2*((*(output+derivative_index))-(*(output_target+derivative_index)));

	return x;
}

double annlCalculateLossTotal (annlSequence sequence)
{
	double loss_total = 0;

	for (int m=0; m<sequence.num_sequence; m++)
	{
		// Set up the input layers.
		for (int i=0; i<sequence.sequence_list[m].num_layer_input; i++)
		{
			memcpy (sequence.sequence_list[m].layer_input_list[i].layer_input->x,
				sequence.sequence_list[m].layer_input_list[i].input_values,
				sizeof(double)*(sequence.sequence_list[m].layer_input_list[i].layer_input->size));
		}

		// Calculate the outputs.
		annlCalculateOutput (sequence.sequence_list[m].layer_start);

		// Save the outputs and determine the total loss.
		for (int i=0; i<sequence.sequence_list[m].num_layer_output; i++)
		{
			memcpy (sequence.sequence_list[m].layer_output_list[i].output_values,
				sequence.sequence_list[m].layer_output_list[i].layer_output->x,
				sizeof(double)*(sequence.sequence_list[m].layer_output_list[i].layer_output->size));

			loss_total += annlCalculateLoss (sequence.sequence_list[m].layer_output_list[i].layer_output->size,
							 sequence.sequence_list[m].layer_output_list[i].output_values,
							 sequence.sequence_list[m].layer_output_list[i].output_target,
							 sequence.sequence_list[m].layer_output_list[i].output_target_fit,
							 NO_DERIVATIVE, 0);
		}
	}

	return loss_total;
}

double annlCalculateLossTotal_omp (annlSequence sequence)
{
	double loss_total_m[sequence.num_sequence];
	double loss_total = 0;

	#pragma omp parallel for
	for (int m=0; m<sequence.num_sequence; m++)
	{
		loss_total_m[m] = 0;

		// Set up the input layers.
		for (int i=0; i<sequence.sequence_list[m].num_layer_input; i++)
		{
			memcpy (sequence.sequence_list[m].layer_input_list[i].layer_input->x,
				sequence.sequence_list[m].layer_input_list[i].input_values,
				sizeof(double)*(sequence.sequence_list[m].layer_input_list[i].layer_input->size));
		}

		// Calculate the outputs.
		annlCalculateOutput (sequence.sequence_list[m].layer_start);

		// Save the outputs and determine the total loss.
		for (int i=0; i<sequence.sequence_list[m].num_layer_output; i++)
		{
			memcpy (sequence.sequence_list[m].layer_output_list[i].output_values,
				sequence.sequence_list[m].layer_output_list[i].layer_output->x,
				sizeof(double)*(sequence.sequence_list[m].layer_output_list[i].layer_output->size));

			loss_total_m[m] += annlCalculateLoss (sequence.sequence_list[m].layer_output_list[i].layer_output->size,
							      sequence.sequence_list[m].layer_output_list[i].output_values,
							      sequence.sequence_list[m].layer_output_list[i].output_target,
							      sequence.sequence_list[m].layer_output_list[i].output_target_fit,
							      NO_DERIVATIVE, 0);
		}
	}

	for (int m=0; m<sequence.num_sequence; m++)
	{
		loss_total += loss_total_m[m];
	}

	return loss_total;
}

void annlUpdateParameters (annlLayer *layer_input, double step)
{
	int i, j, i_max, j_max, layer_w_index;

	annlLayer *layer_previous = layer_input;
	annlLayer *layer_current = layer_previous->layer_next;

	while (layer_current!=NULL)
	{
		i_max = layer_current->size;
		j_max = layer_previous->size;

		for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++) if (layer_current->layer_w[layer_w_index].update_w==1) (*(layer_current->layer_w[layer_w_index].integrate_dw))(layer_current,layer_w_index,step);

		if (layer_current->update_b==1) (*(layer_current->integrate_db))(layer_current,step);

		layer_previous = layer_current;
		layer_current = layer_current->layer_next;
	}

	return;
}

void annlUpdateParameters_omp (annlSequence sequence, double step)
{
	for (int m=1; m<sequence.num_sequence; m++)
	{
		annlLayer *layer_previous_0 = sequence.sequence_list[0].layer_start;
		annlLayer *layer_current_0 = layer_previous_0->layer_next;

		annlLayer *layer_previous = sequence.sequence_list[m].layer_start;
		annlLayer *layer_current = layer_previous->layer_next;

		while (layer_current!=NULL)
		{
			for (int i=0; i<layer_current->b_num; i++) layer_current_0->db[i] += layer_current->db[i];

			for (int layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++)
			{
				for (int i=0; i<layer_current->layer_w[layer_w_index].w_num; i++)
				{
					layer_current_0->layer_w[layer_w_index].dw[i] += layer_current->layer_w[layer_w_index].dw[i];
				}
			}

			layer_previous_0 = layer_current_0;
			layer_current_0 = layer_current_0->layer_next;

			layer_previous = layer_current;
			layer_current = layer_current->layer_next;
		}
	}

	annlLayer *layer_previous = sequence.sequence_list[0].layer_start;
	annlLayer *layer_current = layer_previous->layer_next;

	while (layer_current!=NULL)
	{
		int i_max = layer_current->size;
		int j_max = layer_previous->size;

		for (int layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++) if (layer_current->layer_w[layer_w_index].update_w==1) (*(layer_current->layer_w[layer_w_index].integrate_dw))(layer_current,layer_w_index,step);

		if (layer_current->update_b==1) (*(layer_current->integrate_db))(layer_current,step);

		layer_previous = layer_current;
		layer_current = layer_current->layer_next;
	}

	return;
}

annlLayer* annlCalculateGradient (annlSequence sequence)
{
	int i, j, i_max, j_max, layer_w_index;

	annlLayer *layer_output;
	annlLayer *layer_previous;
	annlLayer *layer_current;

	for (int m=0; m<sequence.num_sequence; m++)
	{
		layer_previous = sequence.sequence_list[m].layer_start;
		layer_current = layer_previous->layer_next;

		while (layer_current!=NULL)
		{
			i_max = layer_current->size;

			for (i=0; i<layer_current->b_num; i++) layer_current->db[i] = 0;

			for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++)
			{
				for (i=0; i<layer_current->layer_w[layer_w_index].w_num; i++)
				{
					layer_current->layer_w[layer_w_index].dw[i] = 0;
				}
			}

			layer_previous = layer_current;
			layer_current = layer_current->layer_next;
		}
	}

	for (int m=0; m<sequence.num_sequence; m++)
	{
		// Set up the input layers.
		for (int i=0; i<sequence.sequence_list[m].num_layer_input; i++)
		{
			memcpy (sequence.sequence_list[m].layer_input_list[i].layer_input->x,
				sequence.sequence_list[m].layer_input_list[i].input_values,
				sizeof(double)*(sequence.sequence_list[m].layer_input_list[i].layer_input->size));
		}

		// Calculate the outputs.
		layer_output = annlCalculateOutput (sequence.sequence_list[m].layer_start);

		layer_current = layer_output;
		layer_previous = layer_current->layer_previous;

		// Set the derivatives of the error with respect to the outputs.
		for (int j=0; j<sequence.sequence_list[m].num_layer_output; j++)
		{
			for (int i=0; i<sequence.sequence_list[m].layer_output_list[j].layer_output->size; i++)
			{
				sequence.sequence_list[m].layer_output_list[j].layer_output->dx[i] = annlCalculateLoss (0,
									                             sequence.sequence_list[m].layer_output_list[j].output_values,
									                             sequence.sequence_list[m].layer_output_list[j].output_target,
									                             sequence.sequence_list[m].layer_output_list[j].output_target_fit,
									                             DERIVATIVE, i);
			}
		}

		while ((layer_current->layer_previous)!=NULL)
		{
			i_max = layer_current->size;
			j_max = layer_previous->size;

			(*(layer_current->activation))(layer_current,DERIVATIVE);

			for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++) (*(layer_current->layer_w[layer_w_index].calc_dw))(layer_current,layer_w_index);

			(*(layer_current->calc_db))(layer_current);

			for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++) (*(layer_current->layer_w[layer_w_index].calc_dxj))(layer_current,layer_w_index);

			// Move back one layer.
			layer_current = layer_previous;
			layer_previous = layer_current->layer_previous;
		}
	}

	return layer_previous;
}

annlLayer* annlCalculateGradient_omp (annlSequence sequence)
{
	int i, j, i_max, j_max, layer_w_index;

	annlLayer *layer_output;
	annlLayer *layer_previous;
	annlLayer *layer_current;

	#pragma omp parallel for private (layer_current, layer_previous, i, i_max, layer_w_index)
	for (int m=0; m<sequence.num_sequence; m++)
	{
		layer_previous = sequence.sequence_list[m].layer_start;
		layer_current = layer_previous->layer_next;

		while (layer_current!=NULL)
		{
			i_max = layer_current->size;

			for (i=0; i<layer_current->b_num; i++) layer_current->db[i] = 0;

			for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++)
			{
				for (i=0; i<layer_current->layer_w[layer_w_index].w_num; i++)
				{
					layer_current->layer_w[layer_w_index].dw[i] = 0;
				}
			}

			layer_previous = layer_current;
			layer_current = layer_current->layer_next;
		}
	}

	#pragma omp parallel for private (layer_output, layer_current, layer_previous, i_max, j_max, layer_w_index)
	for (int m=0; m<sequence.num_sequence; m++)
	{
		// Set up the input layers.
		for (int i=0; i<sequence.sequence_list[m].num_layer_input; i++)
		{
			memcpy (sequence.sequence_list[m].layer_input_list[i].layer_input->x,
				sequence.sequence_list[m].layer_input_list[i].input_values,
				sizeof(double)*(sequence.sequence_list[m].layer_input_list[i].layer_input->size));
		}

		// Calculate the outputs.
		layer_output = annlCalculateOutput (sequence.sequence_list[m].layer_start);

		layer_current = layer_output;
		layer_previous = layer_current->layer_previous;

		// Set the derivatives of the error with respect to the outputs.
		for (int j=0; j<sequence.sequence_list[m].num_layer_output; j++)
		{
			for (int i=0; i<sequence.sequence_list[m].layer_output_list[j].layer_output->size; i++)
			{
				sequence.sequence_list[m].layer_output_list[j].layer_output->dx[i] = annlCalculateLoss (0,
									                             sequence.sequence_list[m].layer_output_list[j].output_values,
									                             sequence.sequence_list[m].layer_output_list[j].output_target,
									                             sequence.sequence_list[m].layer_output_list[j].output_target_fit,
									                             DERIVATIVE, i);
			}
		}

		while ((layer_current->layer_previous)!=NULL)
		{
			i_max = layer_current->size;
			j_max = layer_previous->size;

			(*(layer_current->activation))(layer_current,DERIVATIVE);

			for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++) (*(layer_current->layer_w[layer_w_index].calc_dw))(layer_current,layer_w_index);

			(*(layer_current->calc_db))(layer_current);

			for (layer_w_index=0; layer_w_index<layer_current->num_layer_w; layer_w_index++) (*(layer_current->layer_w[layer_w_index].calc_dxj))(layer_current,layer_w_index);

			// Move back one layer.
			layer_current = layer_previous;
			layer_previous = layer_current->layer_previous;
		}
	}

	return layer_previous;
}

// This is the rectified linear unit.
void annlActivateReLU (annlLayer *layer_current, int derivative)
{
	if (derivative==NO_DERIVATIVE) {for (int i=0; i<layer_current->size; i++) {layer_current->x[i] = (layer_current->z[i]>=0) ? layer_current->z[i] : 0;}}
	else {for (int i=0; i<layer_current->size; i++) {layer_current->dz[i] = (layer_current->z[i]>=0) ? 1 : 0;}}
}

// This is the logistic function.
void annlActivateLogistic (annlLayer *layer_current, int derivative)
{
	if (derivative==NO_DERIVATIVE) {for (int i=0; i<layer_current->size; i++) {layer_current->x[i] = 1/(1+exp(-(layer_current->z[i])));}}
	else {for (int i=0; i<layer_current->size; i++) {layer_current->dz[i] = 1/(4*cosh((layer_current->z[i])/2)*cosh((layer_current->z[i])/2));}}
}

// This is the tanh function.
void annlActivateTanh (annlLayer *layer_current, int derivative)
{
	if (derivative==NO_DERIVATIVE) {for (int i=0; i<layer_current->size; i++) {layer_current->x[i] = tanh(layer_current->z[i]);}}
	else {for (int i=0; i<layer_current->size; i++) {layer_current->dz[i] = 1/(cosh(layer_current->z[i])*cosh(layer_current->z[i]));}}
}

// This is the softmax function.
void annlActivateSoftmax (annlLayer *layer_current, int derivative)
{
	if (derivative==NO_DERIVATIVE)
	{
		double sum = 0;

		for (int i=0; i<layer_current->size; i++) {sum += exp(layer_current->z[i]);}
		for (int i=0; i<layer_current->size; i++) {layer_current->x[i] = exp(layer_current->z[i])/sum;}
	}
	else
	{
		double sum = 0;

		for (int i=0; i<layer_current->size; i++) {sum += exp(layer_current->z[i]);}
		for (int i=0; i<layer_current->size; i++) {layer_current->dz[i] = (sum*exp(layer_current->z[i])-exp(layer_current->z[i])*exp(layer_current->z[i]))/(sum*sum);}
	}
}

// This is the Heaviside step function.
int annlHeavisideTheta (double x) {return x>0 ? 1 : 0;}

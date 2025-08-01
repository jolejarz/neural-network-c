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
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcConvolution_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcConvolution_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcConvolution_dxj;
	if (train==TRAIN_ADAM)
	{
		layer_current->layer_w[layer_w_index].w_m = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].w_v = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateConvolution_dw_Adam;
	}
	else
	{
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateConvolution_dw;
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
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcConvolution_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcConvolution_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcConvolution_dxj;
	layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateConvolution_dw;
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
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcConvolution_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcConvolution_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcConvolution_dxj;
	if (train==TRAIN_ADAM)
	{
		layer_current->layer_w[layer_w_index].w_m = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].w_v = calloc (layer_current->size*layer_previous->size, sizeof(double));
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateConvolution_dw_Adam;
	}
	else
	{
		layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateConvolution_dw;
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
	layer_current->layer_w[layer_w_index].calc_z_w = annlCalcConvolution_z_w;
	layer_current->layer_w[layer_w_index].calc_dw = annlCalcConvolution_dw;
	layer_current->layer_w[layer_w_index].calc_dxj = annlCalcConvolution_dxj;
	layer_current->layer_w[layer_w_index].integrate_dw = annlIntegrateConvolution_dw;
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

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

void annlSetBiasConvolution (annlLayer *layer_current, int L, int n, int train)
{
	// Set the total number of bias parameters.
	layer_current->b_num = n;

	layer_current->b_value = malloc (sizeof(double)*layer_current->size);
	layer_current->db = malloc (sizeof(double)*layer_current->size);
	layer_current->update_b = 1;
	layer_current->calc_z_b = annlCalcConvolution_z_b;
	layer_current->calc_db = annlCalcConvolution_db;
	if (train==TRAIN_ADAM)
	{
		layer_current->b_m = calloc (layer_current->size, sizeof(double));
		layer_current->b_v = calloc (layer_current->size, sizeof(double));
		layer_current->integrate_db = annlIntegrateConvolution_db_Adam;
	}
	else
	{
		layer_current->integrate_db = annlIntegrateConvolution_db;
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

void annlSetBiasConvolutionExisting_b (annlLayer *layer_current, int L, int n, double *b)
{
	// Set the total number of bias parameters.
	layer_current->b_num = n;

	layer_current->b_value = b;
	layer_current->db = malloc (sizeof(double)*layer_current->size);
	layer_current->update_b = 0;
	layer_current->calc_z_b = annlCalcConvolution_z_b;
	layer_current->calc_db = annlCalcConvolution_db;
	layer_current->integrate_db = annlIntegrateConvolution_db;
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

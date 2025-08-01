void annlCalcFull_z_b (annlLayer *layer_current)
{
	int i_max = layer_current->size;

	for (int i=0; i<i_max; i++)
	{
		layer_current->z[i] += layer_current->b[i];
	}

	return;
}

void annlCalcConvolution_z_b (annlLayer *layer_current)
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

void annlCalcConvolution_z_w (annlLayer *layer_current, int layer_w_index)
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

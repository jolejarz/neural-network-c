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

void annlIntegrateConvolution_db (annlLayer *layer_current, double step)
{
	for (int k=0; k<layer_current->b_num; k++)
	{
		layer_current->b_value[k] -= step * layer_current->db[k];
	}

	return;
}

void annlIntegrateConvolution_db_Adam (annlLayer *layer_current, double step)
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

void annlIntegrateConvolution_dw (annlLayer *layer_current, int layer_w_index, double step)
{
	for (int k=0; k<layer_current->layer_w[layer_w_index].w_num; k++)
	{
		layer_current->layer_w[layer_w_index].w_value[k] -= step * layer_current->layer_w[layer_w_index].dw[k];
	}

	return;
}

void annlIntegrateConvolution_dw_Adam (annlLayer *layer_current, int layer_w_index, double step)
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

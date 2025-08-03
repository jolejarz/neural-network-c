void annlCalculateGradient (annlSequence sequence, int batch_size, int b[])
{
	int i, j, i_max, j_max, layer_w_index;

	annlLayer *layer_output;
	annlLayer *layer_previous;
	annlLayer *layer_current;

	for (int m=0; m<batch_size; m++)
	{
		layer_previous = sequence.sequence_list[b[m]].layer_start;
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

	for (int m=0; m<batch_size; m++)
	{
		// Set up the input layers.
		for (int i=0; i<sequence.sequence_list[b[m]].num_layer_input; i++)
		{
			memcpy (sequence.sequence_list[b[m]].layer_input_list[i].layer_input->x,
				sequence.sequence_list[b[m]].layer_input_list[i].input_values,
				sizeof(double)*(sequence.sequence_list[b[m]].layer_input_list[i].layer_input->size));
		}

		// Calculate the outputs.
		layer_output = annlCalculateOutput (sequence.sequence_list[b[m]].layer_start);

		layer_current = layer_output;
		layer_previous = layer_current->layer_previous;

		// Set the derivatives of the error with respect to the outputs.
		for (int j=0; j<sequence.sequence_list[b[m]].num_layer_output; j++)
		{
			for (int i=0; i<sequence.sequence_list[b[m]].layer_output_list[j].layer_output->size; i++)
			{
				sequence.sequence_list[b[m]].layer_output_list[j].layer_output->dx[i] = annlCalculateLoss (0,
									                                sequence.sequence_list[b[m]].layer_output_list[j].output_values,
									                                sequence.sequence_list[b[m]].layer_output_list[j].output_target,
									                                sequence.sequence_list[b[m]].layer_output_list[j].output_target_fit,
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

void annlCalculateGradient_omp (annlSequence sequence)
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

void annlCalcFull_db (annlLayer *layer_current)
{
	int i_max = layer_current->size;

	for (int i=0; i<i_max; i++)
	{
		layer_current->db[i] += layer_current->dx[i] * layer_current->dz[i];
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

void annlCalcConvolution_db (annlLayer *layer_current)
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

void annlCalcConvolution_dw (annlLayer *layer_current, int layer_w_index)
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

void annlCalcConvolution_dxj (annlLayer *layer_current, int layer_w_index)
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

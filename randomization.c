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

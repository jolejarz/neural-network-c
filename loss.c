double annlCalculateLossSquaredError (int output_size, double *output, double *output_target, double *output_target_fit, int derivative, int derivative_index)
{
	double x=0;

	if (derivative==NO_DERIVATIVE)
	{
		for (int i=0; i<output_size; i++)
		{
			if (output_target_fit[i]==1) x += (output[i]-output_target[i]) * (output[i]-output_target[i]);
		}
	}
	else if (output_target_fit[derivative_index]==1) x = 2*(output[derivative_index]-output_target[derivative_index]);

	return x;
}

double annlCalculateLossCrossEntropy (int output_size, double *output, double *output_target, double *output_target_fit, int derivative, int derivative_index)
{
	double x=0;

	if (derivative==NO_DERIVATIVE)
	{
		for (int i=0; i<output_size; i++)
		{
			if (output_target_fit[i]==1) x -= output_target[i] * log(output[i]);
		}
	}
	else if (output_target_fit[derivative_index]==1) x = -output_target[derivative_index]/output[derivative_index];

	return x;
}

double annlCalculateLossTotal (annlSequence sequence, double (*loss_function)(int,double*,double*,double*,int,int))
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

			loss_total += (*loss_function) (sequence.sequence_list[m].layer_output_list[i].layer_output->size,
							sequence.sequence_list[m].layer_output_list[i].output_values,
							sequence.sequence_list[m].layer_output_list[i].output_target,
							sequence.sequence_list[m].layer_output_list[i].output_target_fit,
							NO_DERIVATIVE, 0);
		}
	}

	return loss_total;
}

double annlCalculateLossTotal_omp (annlSequence sequence, double (*loss_function)(int,double*,double*,double*,int,int))
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

			loss_total_m[m] += (*loss_function) (sequence.sequence_list[m].layer_output_list[i].layer_output->size,
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

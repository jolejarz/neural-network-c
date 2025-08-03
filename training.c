void annlTrain (annlSequence sequence, annlLayer *layer_input, double loss_diff, int batch_size, gsl_rng *rng, double step, void (*status)(int,double))
{
	int epoch = 0;
	double loss_test;

	int a[sequence.num_sequence];
	int b[batch_size];
	int b_index;
	int count;
	int batch_count;

	double x;

	// Check if the total loss is greater than or equal to loss_diff.
	while ( (loss_test=annlCalculateLossTotal (sequence)) >= loss_diff )
	{
		// Print the status.
		(*status) (epoch++, loss_test);

		// Set up the array for selecting sequences.
		for (int i=0; i<sequence.num_sequence; i++) a[i]=i;

		// When count is equal to zero, the epoch is complete.
		count = sequence.num_sequence;

		// Perform gradient descent.
		while (count>0)
		{
			// Select batch_size sequences, and begin adding them at index 0;
			batch_count = batch_size;
			b_index = 0;

			// Select a batch.
			while (batch_count>0 && count>0)
			{
				// If a pseudorandom number generator was specified, then use it to select a sequence.
				// Otherwise, select the first sequence in the list.
				x = rng!=NULL ? gsl_rng_uniform (rng) : 0;

				// Select a sequence and add it to the batch.
				b[b_index++] = a[(int)(count*x)];

				// Fill in the hole in the list of sequences to be selected.
				// Decrement the number of sequences that remain to be selected.
				a[(int)(count*x)] = a[(count--)-1];

				// A sequence was just added to the batch, so update batch_count.
				batch_count--;
			}

			// Calculate the gradient.
			annlCalculateGradient (sequence, batch_size, b);

			// Update the parameters.
			annlUpdateParameters (layer_input, step);
		}
	}
	// Print the final status.
	(*status) (epoch, loss_test);

	return;
}

void annlTrain_omp (annlSequence sequence, double loss_diff, double step, void (*status)(int,double))
{
	int epoch = 0;
	double loss_test;

	// Check if the total loss is greater than or equal to loss_diff.
	while ( (loss_test=annlCalculateLossTotal_omp (sequence)) >= loss_diff )
	{
		// Print the status.
		(*status) (epoch++, loss_test);

		// Calculate the gradient.
		annlCalculateGradient_omp (sequence);

		// Update the parameters.
		annlUpdateParameters_omp (sequence, step);
	}
	// Print the final status.
	(*status) (epoch, loss_test);

	return;
}

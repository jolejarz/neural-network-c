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

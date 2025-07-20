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

void annlLinkSequence (annlLayer *layer_previous, annlLayer *layer_next)
{
	layer_previous->layer_next = layer_next;
	layer_next->layer_previous = layer_previous;

	return;
}

// Artificial Neural Network Library for C

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <omp.h>
#include "annl.h"

#include "activation.c"
#include "bias.c"
#include "connection.c"
#include "gradient.c"
#include "integration.c"
#include "layer.c"
#include "loss.c"
#include "output.c"
#include "randomization.c"
#include "sequence.c"

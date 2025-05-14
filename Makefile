CC=gcc
end=-w -fopenmp -lm -lgsl -lgslcblas -O3

all: xor seven-segment_display LeNet5 LeNet5_omp bit_delay

xor: xor.c annl.o
	${CC} -o xor xor.c annl.o ${end}

seven-segment_display: seven-segment_display.c annl.o
	${CC} -o seven-segment_display seven-segment_display.c annl.o ${end}

LeNet5: LeNet5.c annl.o
	${CC} -o LeNet5 LeNet5.c annl.o ${end}

LeNet5_omp: LeNet5_omp.c annl.o
	${CC} -o LeNet5_omp LeNet5_omp.c annl.o ${end}

bit_delay: bit_delay.c annl.o
	${CC} -o bit_delay bit_delay.c annl.o ${end}

annl.o: annl.c
	${CC} -o annl.o -c annl.c ${end}


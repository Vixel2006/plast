CC     = gcc
CFLAGS = -O3 -Wall -march=native -I../include
LDFLAGS = -L.. -lplast -lm -lgomp -fopenmp

EXAMPLES = xor rl_policy_network conv_mnist

.PHONY: all clean

all: $(EXAMPLES)

%: %.c ../libplast.a
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(EXAMPLES)

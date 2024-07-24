CC=nvcc
CFLAGS=-O3
CFLAGS+=`pkg-config --cflags numa`
LDFLAGS=-lpthread -lnuma
LDFLAGS+=`pkg-config --libs numa`

EXE=numa_explicit.exe

all : $(EXE)

%.o : %.c
	$(CC) -c $(CFLAGS) -o $@ $^

$(EXE) : numa_explicit.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

proper :
	rm -f *.o

clean : proper
	rm -f $(EXE)

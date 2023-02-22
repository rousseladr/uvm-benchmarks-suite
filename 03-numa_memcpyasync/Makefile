CC=nvcc
CFLAGS=-O3
LDFLAGS=-lpthread -lnuma

EXE=numa_memcpy-async.exe

all : $(EXE)

%.o : %.c
	$(CC) -c $(CFLAGS) -o $@ $^

$(EXE) : numa_memcpy-async.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

proper :
	rm -f *.o

clean : proper
	rm -f $(EXE)

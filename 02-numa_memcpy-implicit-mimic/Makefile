CC=nvcc
CFLAGS=-O3
LDFLAGS=-lpthread -lnuma

EXE=numa_implicit-mimic.exe

all : $(EXE)

%.o : %.c
	$(CC) -c $(CFLAGS) -o $@ $^

$(EXE) : numa_implicit-mimic.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

proper :
	rm -f *.o

clean : proper
	rm -f $(EXE)

CC=hipcc
CFLAGS=-O3
CFLAGS+=`pkg-config --cflags numa`
LDFLAGS=-lpthread -lnuma
LDFLAGS+=`pkg-config --libs numa`

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

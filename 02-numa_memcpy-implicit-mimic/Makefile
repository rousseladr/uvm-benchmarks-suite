CC=hipcc
CFLAGS=-O3
CFLAGS+=`pkg-config --cflags numa`
LDFLAGS=-lpthread -lnuma
LDFLAGS+=`pkg-config --libs numa`

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

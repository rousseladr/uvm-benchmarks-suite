#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>
#include <hip/hip_runtime.h>
#include <time.h>
#include <inttypes.h>
#include <sys/mman.h>

constexpr int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            fprintf(stderr, "An error encountered: \" %s \" at %s:%d\n", hipGetErrorString(error), __FILE__ ,__LINE__);                          \
            exit(error_exit_code);                                                     \
        }                                                                                   \
    }

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)
/** return time in second
*/
double get_elapsedtime(void)
{
  struct timespec st;
  int err = gettime(&st);
  if (err !=0) return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

/*!
  * Function used to eliminate redundant cpus (i.e. hyperthreads) from a list of cpus
  * @param[in] num_cpus         number of cpus to test
  * @param[in] cpus             list of cpus to test 
  * @param[out] nb_phys_cpus    number of physical cpus, without hyperthreads 
  * @return                     List of physical cpus
 */
int* eliminate_hyperthreads(int num_cpus, int* cpus, int* nb_phys_cpus)
{
  bool* smt = (bool*)malloc(sizeof(bool) * CPU_SETSIZE);
  memset((void*)smt, 0, sizeof(bool) * CPU_SETSIZE);

  FILE* input;
  *nb_phys_cpus = 0;
  // This loop eliminates the hyper-threads to only conserve one PU per processor
  for(int i = 0; i < num_cpus; ++i)
  {
    char input_file[1024];
    sprintf(input_file, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpus[i]);
    input = fopen(input_file, "r");
    if(input == NULL)
    {
      perror("fopen");
      exit(EXIT_FAILURE);
    }
    char* line = NULL;
    size_t len;
    // Get the list of SMT on the cpu cpus[i]
    // Format gives list of PU separated by commas
    if(getline(&line, &len, input) == -1)
    {
      perror("getline");
      exit(EXIT_FAILURE);
    }

    // We are only interested by the first PU in the list,
    // so find the 1st occurence of the comma
    char* delim = strpbrk(line, ",");
    if(delim != NULL)
    {
      // Ends the line by filling it with '\0'
      // and read the value before ',' symbol
      *delim = '\0';
    }

    // Convert into integer
    int cur_cpu = atoi(line);

    // if cur_cpu has already been set (false in smt array), then continue
    // Else, set the pu to true
    if(!smt[cur_cpu] && cur_cpu >= 0 && cur_cpu < CPU_SETSIZE)
    {
      *nb_phys_cpus += 1;
      smt[cur_cpu] = true;
    }
    fclose(input);
    if(line != NULL)
    {
      free(line);
    }
  }
  int* phys_cpus = (int*) malloc(sizeof(int) * (*nb_phys_cpus));

  int li = 0;
  for(int i = 0; i < CPU_SETSIZE; ++i)
  {
    if(smt[i])
    {
      phys_cpus[li] = i;
      li++;
    }
  }
  free(smt);

  return phys_cpus;
}

void pinThread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_t current_thread = pthread_self();
  if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &set) != 0)
  {
    perror("pthread_setaffinity_np");
    exit(EXIT_FAILURE);
  }
}

//#define N 1E8

#define handle_error_en(en, msg) \
  do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

__global__ void copy( uint64_t* dst, uint64_t* src, size_t n )
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x * gridDim.x )
    {
        dst[i] = src[i];
    }
}

__global__ void init( uint64_t* dst, uint64_t val, size_t n )
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x * gridDim.x )
    {
        dst[i] = val;
    }
}

int main(int argc, char *argv[])
{
  int nb_test = 25;
  int s, j;
  int cpu = -1;
  uint64_t size_in_mbytes = 100;
  bool verbose = false;
  bool device_copy = false;
  bool check = false;

  int opt;
  while ((opt = getopt(argc, argv, "vhs:i:dc")) != -1)
  {
    switch (opt)
    {
      case 's':
        size_in_mbytes = (uint64_t)atoi(optarg);
        break;
      case 'i':
        nb_test = (int)atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'c':
        check = true;
        break;
      case 'd':
	device_copy = true;
	break;
      case 'h':
        goto usage;
        break;
      default:
        goto usage;
    }
  }

  if (optind != argc)
  {
usage:
    fprintf(stdout, "CUDA Bench - Async. Memory Transfers Throughput evaluation with NUMA consideration 1.0.0\n");
    fprintf(stdout, "usage: numa_memcpy-async.exe\n\t[-s size in MB]\n\t[-h print this help]\n");
    fprintf(stdout, "\nPlot results using python3:\n");
    fprintf(stdout, "numa_memcpy-async.exe -s <arg> && python3 plot.py <arg>\n");
    exit(EXIT_SUCCESS);
  }

  nb_test+=1;
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  int numcores = sysconf(_SC_NPROCESSORS_ONLN); // divided by 2 because of hyperthreading
  int numanodes = numa_num_configured_nodes();

  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0)
  {
    perror("sched_getaffinity");
    exit(EXIT_FAILURE);
  }

  // enumerate available CPUs
  int* cpus = (int*)malloc(sizeof(int) * numcores);
  int li=0;
  for (int i = 0; i < CPU_SETSIZE; ++i)
  {
    if (CPU_ISSET(i, &set))
    {
      cpus[li] = i;
      li++;
    }
  }
  numcores = li;

  int nb_phys_cpus = 0;
  int* phys_cpus = eliminate_hyperthreads(numcores, cpus, &nb_phys_cpus);

  free(cpus);

  numcores = nb_phys_cpus;
  cpus = phys_cpus;

  int gpucount = -1;
  HIP_CHECK(hipGetDeviceCount(&gpucount));

  double duration = 0.;
  int *tgpu = (int*)malloc(sizeof(int) * numcores * gpucount);
  double *HtD = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *DtH = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *HtD_gbs = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *DtH_gbs = (double*)malloc(sizeof(double) * numcores * gpucount);
  memset(tgpu, -1, sizeof(int) * numcores * gpucount);
  memset(HtD, 0, sizeof(double) * numcores * gpucount);
  memset(DtH, 0, sizeof(double) * numcores * gpucount);
  memset(HtD_gbs, 0, sizeof(double) * numcores * gpucount);
  memset(DtH_gbs, 0, sizeof(double) * numcores * gpucount);

  double size_in_kbytes = size_in_mbytes*1000;
  double size_in_bytes = size_in_kbytes*1000;

  if(verbose)
  {
#ifdef DEBUG
    fprintf(stdout, "Size of array: %lu Bytes\n", (uint64_t)(size_in_bytes));
    fprintf(stdout, "Size of array: %.2f KB\n", (double)(size_in_kbytes));
#endif
    fprintf(stdout, "Size of array: %.2f MB\n", (double)(size_in_mbytes));

#ifdef DISPLAY_BITS
    float size_kb = (float)(size_in_kbytes * CHAR_BIT);
    float size_mb = (float)(size_in_mbytes * CHAR_BIT);
    fprintf(stdout, "Size of array: %lu bits\n", (uint64_t)(size_in_bytes * CHAR_BIT));
    fprintf(stdout, "Size of array: %.2f Kb\n", size_kb);
    fprintf(stdout, "Size of array: %.2f Mb\n", size_mb);
#endif
  }

  uint64_t N = (size_in_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

#ifdef DEBUG
  if(verbose)
  {
    fprintf(stdout, "N = %lu\n", N);
  }
#endif


  for (int i = 0; i < numcores; ++i)
  {
    int coreId = cpus[i];

    if(coreId < 0 || coreId >= numcores)
    {
      fprintf(stdout, "FATAL ERROR! Invalid core id\n");
      exit(EXIT_FAILURE);
    }

    if(verbose)
    {
      fprintf(stdout, "Target core %d\n", coreId);
    }
    /* Set affinity mask to include CPUs coreId */
    pinThread(coreId);

    int cur_numanode = numa_node_of_cpu(cpu);
    if(verbose)
    {
      fprintf(stdout, "Running on CPU %d of %d\n", cpu, numcores);
      fprintf(stdout, "Running on NUMA %d of %d\n", cur_numanode, numanodes);
    }

    for(int deviceId = 0; deviceId < gpucount; ++deviceId)
    {
      HIP_CHECK(hipSetDevice(deviceId));
      if(verbose)
      {
        fprintf(stdout, "Set Device to %d\n", deviceId);
      }
      tgpu[i * gpucount + deviceId] = deviceId;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      hipEvent_t start, stop;
      HIP_CHECK(hipEventCreate(&start));
      HIP_CHECK(hipEventCreate(&stop));

      uint64_t *A;
      HIP_CHECK(hipHostMalloc(&A, N * sizeof(uint64_t)));

      for(int k = 0 ; k < N; ++k)
      {
        A[k] = k;
      }

      //int allocnumaid = -1;
      //get_mempolicy(&allocnumaid, NULL, 0, (void*)A, MPOL_F_NODE | MPOL_F_ADDR);
      //if(allocnumaid != cur_numanode)
      //{
      //  fprintf(stderr, "ERROR: bad NUMA allocation\n");
      //  hipHostFree(A);
      //  free(tgpu);
      //  free(HtD);
      //  free(DtH);
      //  free(HtD_gbs);
      //  free(DtH_gbs);
      //  exit(EXIT_FAILURE);
      //}

      uint64_t *d_A;
      HIP_CHECK(hipMalloc(&d_A, N * sizeof(uint64_t)));

      duration = 0.;
      double throughput = 0.;
      double t0 = 0., t1 = 0.;

      dim3  dimBlock(64, 1, 1);
      dim3  dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);

      HIP_CHECK(hipDeviceSynchronize());
      for(int k = 0; k < nb_test; ++k)
      {
        HIP_CHECK(hipStreamSynchronize(stream));

	if(check)
	{
		for(int toto = 0; toto < N; ++toto) A[toto] = (uint64_t)k;
	  if(check && (A[0] != (uint64_t)k))
	  {
	    fprintf(stderr, "BAD INIT!!\n");
	    exit(-1);
	  }
	}

	t0 = get_elapsedtime();
        HIP_CHECK(hipMemcpyAsync(d_A, A, N * sizeof(uint64_t), hipMemcpyHostToDevice, stream));
	HIP_CHECK(hipStreamSynchronize(stream));
	t1 = get_elapsedtime();

	if(k == 0) { continue; }
        duration += (t1 - t0);

#ifdef DEBUG
        get_mempolicy(&allocnumaid, NULL, 0, (void*)A, MPOL_F_NODE | MPOL_F_ADDR);
        if(allocnumaid != cur_numanode)
        {
          fprintf(stderr, "FATAL ERROR!!\n");
          exit(-1);
        }
#endif
      }

      duration /= nb_test-1;
      throughput = size_in_mbytes / (duration * 1000);
      if(verbose)
      {
        fprintf(stdout, "Performance results: \n");
        fprintf(stdout, "HostToDevice>  Time: %lf s\n", duration);
        fprintf(stdout, "HostToDevice>  Throughput: %.2lf GB/s\n", throughput);
      }
      HtD[i * gpucount + deviceId] = duration;
      HtD_gbs[i * gpucount + deviceId] = throughput;

      duration = 0.;
      t0 = t1 = 0.;
      HIP_CHECK(hipDeviceSynchronize());
      for(int k = 0; k < nb_test; ++k)
      {
        HIP_CHECK(hipDeviceSynchronize());
	if(check)
	{
	  init<<<dimGrid, dimBlock, 0, hipStreamDefault>>>(d_A, (uint64_t)k, N);
          HIP_CHECK(hipDeviceSynchronize());
	  if(check && (d_A[0] != (uint64_t)k))
	  {
	    fprintf(stderr, "BAD INIT!!\n");
	    exit(-1);
	  }
	}
        HIP_CHECK(hipDeviceSynchronize());
	if(!device_copy)
	{
	  t0 = get_elapsedtime();
          HIP_CHECK(hipMemcpyAsync(A, d_A, N * sizeof(uint64_t), hipMemcpyDeviceToHost, stream));
	  HIP_CHECK(hipStreamSynchronize(stream));
	  t1 = get_elapsedtime();
	}
	else
	{
          t0 = get_elapsedtime(); 
	  copy<<<dimGrid, dimBlock, 0, stream>>>(A, d_A, N);
          HIP_CHECK(hipStreamSynchronize(stream));
	  t1 = get_elapsedtime();
	  if(check)
	  {
	    for(int toto = 0; toto < N; ++toto)
	    {
	            if(A[toto] != uint64_t(k))
	            {
	              fprintf(stderr, "%d BAD TERMINATION!!!\n", toto);
	              exit(-1);
	            }
	    }
	  }
	} 

	if(k == 0) { continue; }
        duration += (t1 - t0);
      }

      duration /= nb_test-1;
      throughput = size_in_mbytes / (duration * 1000);
      if(verbose)
      {
        fprintf(stdout, "DeviceToHost>  Time: %lf s\n", duration);
        fprintf(stdout, "DeviceToHost>  Throughput: %.2lf GB/s\n\n", throughput);
      }
      DtH[i * gpucount + deviceId] = duration;
      DtH_gbs[i * gpucount + deviceId] = throughput;

      HIP_CHECK(hipFree(d_A));
      HIP_CHECK(hipHostFree(A));
    }
  }

  char buff_memcpyasync_time[100];
  snprintf(buff_memcpyasync_time, 100, "%lu-MB_numa_memcpyasync_time.csv", size_in_mbytes);
  FILE * outputFile;
  outputFile = fopen( buff_memcpyasync_time, "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", buff_memcpyasync_time );
    exit(EXIT_FAILURE);
  }

  fprintf(outputFile, "core\tgpu\tHostToDevice\tDeviceToHost\n");
  for(int i = 0; i < numcores; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", cpus[i], tgpu[i * gpucount + d], HtD[i * gpucount + d], DtH[i * gpucount + d]);
    }
  }

  fclose(outputFile);

  char buff_memcpyasync_gbs[100];
  snprintf(buff_memcpyasync_gbs, 100, "%lu-MB_numa_memcpyasync_gbs.csv", size_in_mbytes);
  outputFile = fopen( buff_memcpyasync_gbs, "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", buff_memcpyasync_gbs );
    exit(EXIT_FAILURE);
  }

  fprintf(outputFile, "core\tgpu\tHostToDevice\tDeviceToHost\n");
  for(int i = 0; i < numcores; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", cpus[i], tgpu[i * gpucount + d], HtD_gbs[i * gpucount + d], DtH_gbs[i * gpucount + d]);
    }
  }

  fclose(outputFile);

  fprintf(stdout, "Results saved in:\n\tGB/s: %s\n", buff_memcpyasync_gbs);
  fprintf(stdout, "\tTime: %s\n", buff_memcpyasync_time);

  free(tgpu);
  free(HtD);
  free(DtH);
  free(HtD_gbs);
  free(DtH_gbs);

  exit(EXIT_SUCCESS);
}

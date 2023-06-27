#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <numa.h>
#include <cuda.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>

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

//#define N (unsigned long int)1E6

__global__
void init(uint64_t *x, uint64_t val, uint64_t N)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < N)
    x[index] = val;
}


#define handle_error_en(en, msg) \
  do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

int main(int argc, char *argv[])
{
  int nb_test = 25;
  int s, j;
  int cpu = -1;
  uint64_t size_in_mbytes = 100;
  bool verbose = false;

  int opt;
  while ((opt = getopt(argc, argv, "vhs:i:")) != -1)
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
    fprintf(stdout, "CUDA Bench - Managed (Implicit) Memory Transfers Throughput evaluation with NUMA consideration 1.0.0\n");
    fprintf(stdout, "usage: numa_implicit.exe\n\t[-s size in MB]\n\t[-h print this help]\n");
    fprintf(stdout, "\nPlot results using python3:\n");
    fprintf(stdout, "numa_implicit.exe -s <arg> && python3 plot.py <arg>\n");
    exit(EXIT_SUCCESS);
  }

  nb_test++;
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  int numcores = sysconf(_SC_NPROCESSORS_ONLN);
  int numanodes = numa_num_configured_nodes();

  int gpucount = -1;
  cudaGetDeviceCount(&gpucount);

  //double t0 = 0., t1 = 0., duration = 0.;
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

#define DEBUG
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
    printf("N = %lu\n", N);
    printf("sizeof(uint64_t) = %lu\n", sizeof(uint64_t));
  }
#endif

  int coreId = 0;

  while( coreId < numcores)
  {

    if(coreId < 0 || coreId >= numcores)
    {
      printf("FATAL ERROR! Invalid core id\n");
      exit(EXIT_FAILURE);
    }

    if(verbose)
    {
      printf("Target core %d\n", coreId);
    }
    /* Set affinity mask to include CPUs coreId */

    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);

    s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      handle_error_en(s, "pthread_setaffinity_np");

    /* Check the actual affinity mask assigned to the thread */

    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      handle_error_en(s, "pthread_getaffinity_np");

    for (j = 0; j < CPU_SETSIZE; j++)
    {
      if (CPU_ISSET(j, &cpuset))
      {
        cpu = j;
        break;
      }
    }

    if(j == CPU_SETSIZE)
    {
      printf("FATAL ERROR! Don't know on which core the thread is placed\n");
      exit(EXIT_FAILURE);
    }

    int cur_numanode = numa_node_of_cpu(cpu);
    if(verbose)
    {
      fprintf(stdout, "Running on CPU %d of %d\n", cpu, numcores);
      fprintf(stdout, "Running on NUMA %d of %d\n", cur_numanode, numanodes);
    }

    for(int deviceId = 0; deviceId < gpucount; ++deviceId)
    {
      cudaSetDevice(deviceId);
      if(verbose)
      {
        fprintf(stdout, "Set Device to %d\n", deviceId);
      }
      tgpu[coreId * gpucount + deviceId] = deviceId;

      uint64_t *d_A;
      cudaStream_t stream;

      cudaMallocManaged(&d_A, N * sizeof(uint64_t));
      cudaMemAdvise(d_A, N * sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
      cudaStreamCreate(&stream);

      double t0 = 0., t1 = 0.;
      duration = 0.;
      for(int k = 0; k < nb_test; ++k)
      {
        for(int i = 0 ; i < N; ++i)
        {
          d_A[i] += (uint64_t)i;
        }
        cudaDeviceSynchronize();

	t0 = get_elapsedtime();
        cudaMemPrefetchAsync(d_A, N * sizeof(uint64_t), deviceId, stream);
        cudaStreamSynchronize(stream);
	t1 = get_elapsedtime();

	if(k == 0) { continue; }
	fprintf(stdout, "(%d, %d) iter: %d | time: %lf | gbs: %lf\n", coreId, deviceId, k, (t1 - t0), size_in_mbytes / ((t1-t0)*1000));
        duration += (t1 - t0);
      }

      duration /= nb_test;
      double throughput = size_in_mbytes / (duration * 1000);
      if(verbose)
      {
        fprintf(stdout, "Performance results: \n");
        fprintf(stdout, "HostToDevice>  Time: %lf s\n", duration);
        fprintf(stdout, "HostToDevice>  Throughput: %.2lf GB/s\n", throughput);
      }
      HtD[coreId * gpucount + deviceId] = duration;
      HtD_gbs[coreId * gpucount + deviceId] = throughput;
      cudaFree(d_A);
      cudaStreamSynchronize(stream);

      uint64_t *d_B;
      cudaMallocManaged(&d_B, N * sizeof(uint64_t));
      cudaMemAdvise(d_B, N * sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, deviceId);
      cudaDeviceSynchronize();

      dim3 blockSize(32, 1, 1);
      int nbBlocks = (N + 32 - 1) / 32;
      dim3 gridSize(nbBlocks, 1, 1);

      t0 = 0.; t1 = 0.; duration = 0.;
      for(int k = 0; k < nb_test; ++k)
      {
        // First: push data on GPU
        init<<<gridSize, blockSize>>>(d_B, 0x0, N);
        cudaDeviceSynchronize();

        t0 = get_elapsedtime(); 
        // Second: transfer data from GPU to CPU using prefetch mecanism
        cudaMemPrefetchAsync(d_B, N * sizeof(uint64_t), cudaCpuDeviceId, stream);
        // Wait until completion
        cudaStreamSynchronize(stream);
	t1 = get_elapsedtime();

	if(k == 0) { continue; }
        duration += (t1 - t0);
      }

      cudaFree(d_B);
      cudaStreamDestroy(stream);

      duration /= nb_test;
      throughput = size_in_mbytes / (duration * 1000);
      if(verbose)
      {
        fprintf(stdout, "DeviceToHost>  Time: %lf s\n", duration);
        fprintf(stdout, "DeviceToHost>  Throughput: %.2lf GB/s\n\n", throughput);
      }
      DtH[coreId * gpucount + deviceId] = duration;
      DtH_gbs[coreId * gpucount + deviceId] = throughput;

    }
    coreId += 72;
    if(coreId == 72)
    {
      continue;
    }
    coreId++;
  }

  char buff_implicit_time[100];
  snprintf(buff_implicit_time, 100, "%lu-MB_numa_implicit_time.csv", size_in_mbytes);
  FILE * outputFile;
  outputFile = fopen( buff_implicit_time, "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", buff_implicit_time );
    exit(EXIT_FAILURE);
  }

  fprintf(outputFile, "core\tgpu\tHostToDevice\tDeviceToHost\n");
  for(int i = 0; i < numcores; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", i, tgpu[i * gpucount + d], HtD[i * gpucount + d], DtH[i * gpucount + d]);
    }
  }

  fclose(outputFile);

  char buff_implicit_gbs[100];
  snprintf(buff_implicit_gbs, 100, "%lu-MB_numa_implicit_gbs.csv", size_in_mbytes);
  outputFile = fopen( buff_implicit_gbs, "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", buff_implicit_gbs );
    exit(EXIT_FAILURE);
  }

  fprintf(outputFile, "core\tgpu\tHostToDevice\tDeviceToHost\n");
  for(int i = 0; i < numcores; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", i, tgpu[i * gpucount + d], HtD_gbs[i * gpucount + d], DtH_gbs[i * gpucount + d]);
    }
  }

  fclose(outputFile);

  fprintf(stdout, "Results saved in:\n\tGB/s: %s\n", buff_implicit_gbs);
  fprintf(stdout, "\tTime: %s\n", buff_implicit_time);

  free(tgpu);
  free(HtD);
  free(DtH);
  free(HtD_gbs);
  free(DtH_gbs);

  exit(EXIT_SUCCESS);
}

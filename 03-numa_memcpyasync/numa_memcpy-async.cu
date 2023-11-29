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
#include <cuda.h>
#include <time.h>
#include <inttypes.h>
#include <sys/mman.h>

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

//#define N 1E8

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

  int gpucount = -1;
  cudaGetDeviceCount(&gpucount);

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

  int coreId = 0;

  while( coreId < numcores)
  {

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
      fprintf(stderr, "FATAL ERROR! Don't know on which core the thread is placed\n");
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

      cudaStream_t stream;
      cudaStreamCreate(&stream);

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      uint64_t *A;
      cudaMallocHost(&A, N * sizeof(uint64_t));

      for(int i = 0 ; i < N; ++i)
      {
        A[i] = i;
      }

      int allocnumaid = -1;
      get_mempolicy(&allocnumaid, NULL, 0, (void*)A, MPOL_F_NODE | MPOL_F_ADDR);
      if(allocnumaid != cur_numanode)
      {
        fprintf(stderr, "ERROR: bad NUMA allocation\n");
        cudaFreeHost(A);
        free(tgpu);
        free(HtD);
        free(DtH);
        free(HtD_gbs);
        free(DtH_gbs);
        exit(EXIT_FAILURE);
      }

      uint64_t *d_A;
      cudaMalloc(&d_A, N * sizeof(uint64_t));

      duration = 0.;
      double throughput = 0.;
      double t0 = 0., t1 = 0.;
      cudaDeviceSynchronize();
      for(int k = 0; k < nb_test; ++k)
      {
        cudaStreamSynchronize(stream);

	t0 = get_elapsedtime();
        cudaMemcpyAsync(d_A, A, N * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
	cudaStreamSynchronize(stream);
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
      HtD[coreId * gpucount + deviceId] = duration;
      HtD_gbs[coreId * gpucount + deviceId] = throughput;

      duration = 0.;
      t0 = t1 = 0.;
      cudaDeviceSynchronize();
      for(int k = 0; k < nb_test; ++k)
      {
        cudaStreamSynchronize(stream);

	t0 = get_elapsedtime();
        cudaMemcpyAsync(A, d_A, N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	t1 = get_elapsedtime();

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
      DtH[coreId * gpucount + deviceId] = duration;
      DtH_gbs[coreId * gpucount + deviceId] = throughput;

      cudaFree(d_A);
      cudaFreeHost(A);
      //coreId += numcores / numanodes;
    }
    coreId++;
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
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", i, tgpu[i * gpucount + d], HtD[i * gpucount + d], DtH[i * gpucount + d]);
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
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", i, tgpu[i * gpucount + d], HtD_gbs[i * gpucount + d], DtH_gbs[i * gpucount + d]);
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

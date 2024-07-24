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
    fprintf(stdout, "CUDA Bench - Explicit Memory Transfers Throughput evaluation with NUMA consideration 1.0.0\n");
    fprintf(stdout, "usage: numa_explicit.exe\n\t[-s size in MB]\n\t[-h print this help]\n");
    fprintf(stdout, "\nPlot results using python3:\n");
    fprintf(stdout, "numa_explicit.exe -s <arg> && python3 plot.py <arg>\n");
    exit(EXIT_SUCCESS);
  }

  nb_test++;

  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  int gpucount = -1;
  HIP_CHECK(hipGetDeviceCount(&gpucount));

  double duration = 0.;
  int *tgpu = (int*)malloc(sizeof(int) * gpucount * gpucount);
  double *DtD = (double*)malloc(sizeof(double) * gpucount * gpucount);
  double *DtD_gbs = (double*)malloc(sizeof(double) * gpucount * gpucount);
  double *DtDsm = (double*)malloc(sizeof(double) * gpucount * gpucount);
  double *DtDsm_gbs = (double*)malloc(sizeof(double) * gpucount * gpucount);
  memset(tgpu, -1, sizeof(int) * gpucount * gpucount);
  memset(DtD, 0, sizeof(double) * gpucount * gpucount);
  memset(DtD_gbs, 0, sizeof(double) * gpucount * gpucount);
  memset(DtDsm, 0, sizeof(double) * gpucount * gpucount);
  memset(DtDsm_gbs, 0, sizeof(double) * gpucount * gpucount);

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

  int gpu_src = 0;

  while( gpu_src < gpucount)
  {

    if(gpu_src < 0 || gpu_src >= gpucount)
    {
      fprintf(stdout, "FATAL ERROR! Invalid device id (#%d)\n", gpu_src);
      exit(EXIT_FAILURE);
    }

    if(verbose)
    {
      fprintf(stdout, "Target device %d\n", gpu_src);
    }

    uint64_t *A;
    A = (uint64_t*) mmap(0, N * sizeof(uint64_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

    for(int i = 0 ; i < N; ++i)
    {
      A[i] = i;
    }

    HIP_CHECK(hipSetDevice(gpu_src));
    uint64_t *d_src;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(d_src, A, N * sizeof(uint64_t), hipMemcpyHostToDevice));

    for(int gpu_dest = 0; gpu_dest < gpucount; ++gpu_dest)
    {
      HIP_CHECK(hipSetDevice(gpu_dest));
      if(verbose)
      {
        fprintf(stdout, "Set Device to %d\n", gpu_dest);
      }
      tgpu[gpu_src * gpucount + gpu_dest] = gpu_dest;

      uint64_t *d_dest;
      HIP_CHECK(hipMalloc(&d_dest, N * sizeof(uint64_t)));

      double t0 = 0.;
      double t1 = 0.;
      duration = 0.;
      double throughput = 0.;

      dim3  dimBlock(64, 1, 1);
      dim3  dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);

      HIP_CHECK(hipDeviceSynchronize());
      for(int k = 0; k < nb_test; ++k)
      {

	HIP_CHECK(hipDeviceSynchronize());
	t0 = get_elapsedtime();
        HIP_CHECK(hipMemcpyAsync(d_dest, d_src, N * sizeof(uint64_t), hipMemcpyDeviceToDevice, 0));
        HIP_CHECK(hipStreamSynchronize(0));
	t1 = get_elapsedtime();

	if(k == 0) { continue; }
	if(verbose)
	{
	  fprintf(stdout, "(%d, %d) iter: %d | time: %lf | gbs: %lf\n", gpu_src, gpu_dest, k, (t1 - t0), size_in_mbytes / ((t1-t0)*1000));
	}
        duration += (t1 - t0);
      }
      duration /= nb_test-1;

      throughput = size_in_mbytes / (duration * 1000);
      if(verbose)
      {
        fprintf(stdout, "Performance results: \n");
        fprintf(stdout, "CE_DeviceToDevice>  Time: %lf s\n", duration);
        fprintf(stdout, "CE_DeviceToDevice>  Throughput: %.2lf GB/s\n", throughput);
      }
      DtD[gpu_src * gpucount + gpu_dest] = duration;
      DtD_gbs[gpu_src * gpucount + gpu_dest] = throughput;

      duration = 0.;

      HIP_CHECK(hipDeviceSynchronize());
      for(int k = 0; k < nb_test; ++k)
      {
        t0 = get_elapsedtime(); 
	copy<<<dimGrid, dimBlock, 0, hipStreamDefault>>>(d_dest, d_src, N);
        HIP_CHECK(hipStreamSynchronize(0));
	t1 = get_elapsedtime();

	if(k == 0) { continue; }
	if(verbose)
	{
	  fprintf(stdout, "(%d, %d) iter: %d | time: %lf | gbs: %lf\n", gpu_src, gpu_dest, k, (t1 - t0), size_in_mbytes / ((t1-t0)*1000));
	}
        duration += (t1 - t0);
      }

      duration /= nb_test-1;

      throughput = size_in_mbytes / (duration * 1000);
      if(verbose)
      {
        fprintf(stdout, "Performance results: \n");
        fprintf(stdout, "SM_DeviceToDevice>  Time: %lf s\n", duration);
        fprintf(stdout, "SM_DeviceToDevice>  Throughput: %.2lf GB/s\n", throughput);
      }
      DtDsm[gpu_src * gpucount + gpu_dest] = duration;
      DtDsm_gbs[gpu_src * gpucount + gpu_dest] = throughput;

      HIP_CHECK(hipFree(d_dest));
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipFree(d_src));
    munmap(A, N * sizeof(uint64_t));
    gpu_src++;
  }

  char buff_explicit_time[100];
  snprintf(buff_explicit_time, 100, "%lu-MB_numa_explicit_time.csv", size_in_mbytes);
  FILE * outputFile;
  outputFile = fopen( buff_explicit_time, "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", buff_explicit_time );
    exit(EXIT_FAILURE);
  }

  fprintf(outputFile, "core\tgpu\tDeviceToDevice_CopyEngine\tDeviceToDevice_SM\n");
  for(int i = 0; i < gpucount; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", i, tgpu[i * gpucount + d], DtD[i * gpucount + d], DtDsm[i * gpucount + d]);
    }
  }

  fclose(outputFile);

  char buff_explicit_gbs[100];
  snprintf(buff_explicit_gbs, 100, "%lu-MB_numa_explicit_gbs.csv", size_in_mbytes);
  outputFile = fopen( buff_explicit_gbs, "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", buff_explicit_gbs );
    exit(EXIT_FAILURE);
  }

  fprintf(outputFile, "core\tgpu\tDeviceToDevice_CopyEngine\tDeviceToDevice_SM\n");
  for(int i = 0; i < gpucount; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", i, tgpu[i * gpucount + d], DtD_gbs[i * gpucount + d],DtDsm_gbs[i * gpucount + d]);
    }
  }

  fclose(outputFile);

  fprintf(stdout, "Results saved in:\n\tGB/s: %s\n", buff_explicit_gbs);
  fprintf(stdout, "\tTime: %s\n", buff_explicit_time);

  free(tgpu);
  free(DtD);
  free(DtD_gbs);

  exit(EXIT_SUCCESS);
}

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

#define N 1E6

#define handle_error_en(en, msg) \
  do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

int main(int argc, char *argv[])
{
  const int nb_test = 20;
  int s, j;
  int cpu = -1;
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  int numcores = sysconf(_SC_NPROCESSORS_ONLN) / 2; // divided by 2 because of hyperthreading
  int numanodes = numa_num_configured_nodes();

  int gpucount = -1;
  cudaGetDeviceCount(&gpucount);

  double t0 = 0., t1 = 0., duration = 0.;
  int *tgpu = (int*)malloc(sizeof(int) * numcores * gpucount);
  double *HtD = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *DtH = (double*)malloc(sizeof(double) * numcores * gpucount);
  memset(tgpu, -1, sizeof(double) * numcores);
  memset(HtD, 0, sizeof(double) * numcores);
  memset(DtH, 0, sizeof(double) * numcores);

  int coreId = 0;

  while( coreId < numcores)
  {

    if(coreId < 0 || coreId >= numcores)
    {
      printf("FATAL ERROR! Invalid core id\n");
      exit(EXIT_FAILURE);
    }

    printf("Target core %d\n", coreId);
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
    printf("Running on CPU %d of %d\n", cpu, numcores);
    printf("Running on NUMA %d of %d\n", cur_numanode, numanodes);

    //int deviceId = (cur_numanode/2)%gpucount;
    //int deviceId = coreId%gpucount;
    //int deviceId = 0;
    for(int deviceId = 0; deviceId < gpucount; ++deviceId)
    {
      cudaSetDevice(deviceId);
      printf("Set Device to %d\n", deviceId);
      tgpu[coreId * gpucount + deviceId] = deviceId;

      double *A;
      A = (double*) malloc(N * sizeof(double));

      for(int i = 0 ; i < N; ++i)
      {
        A[i] = 1.0 * i;
      }

      double *d_A;
      cudaMalloc(&d_A, N * sizeof(double));

      duration = 0.;
      for(int k = 0; k < nb_test; ++k)
      {
        t0 = get_elapsedtime();

        cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        t1 = get_elapsedtime();
        duration += (t1 - t0);
      }

      duration /= nb_test;
      fprintf(stdout, "Performance results: \n");
      fprintf(stdout, "HostToDevice>  Time: %lf s\n", duration);
      HtD[coreId * gpucount + deviceId] = duration;

      duration = 0.;
      for(int k = 0; k < nb_test; ++k)
      {
        t0 = get_elapsedtime();

        cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        t1 = get_elapsedtime();

        duration += (t1 - t0);
      }

      duration /= nb_test;
      fprintf(stdout, "DeviceToHost>  Time: %lf s\n\n", duration);
      DtH[coreId * gpucount + deviceId] = duration;

      cudaFree(d_A);
      free(A);
      //coreId += numcores / numanodes;
    }
    coreId += 1;
  }

  FILE * outputFile;
  outputFile = fopen( "numa_explicit.csv", "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", "numa_explicit.csv" );
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

  free(HtD);
  free(DtH);

  exit(EXIT_SUCCESS);
}

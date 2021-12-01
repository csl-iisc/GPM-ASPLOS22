#include <thread>
#include <nvml.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include "gpm-annotations.cuh"
#include <sys/wait.h>

using namespace std; 
#define MONITOR_BW
#ifdef MONITOR_BW

static bool FINISH_BW_MONITOR = false; 
static std::thread BW_MONITOR_THREAD;
static std::thread BW_MONITOR_THREAD_2;

//static bool FINISH_WA_MONITOR = false; 
//static std::thread WA_MONITOR_THREAD;

using clock_value_t = long long;
static long long child_pid; 

static __device__ void sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock() - start; }
    while (cycles_elapsed < sleep_cycles);
}

static void bw_stat(const char *str = "bw_stats.csv") {
    FILE *fptr; 
    fptr = fopen(str, "w");
    nvmlInit();
    nvmlDevice_t tester;
    nvmlDeviceGetHandleByIndex (0, &tester);
    unsigned int a, b;
    while(!FINISH_BW_MONITOR) {
        int exec = nvmlDeviceGetPcieThroughput(tester, NVML_PCIE_UTIL_TX_BYTES, &a);
        int exec2 = nvmlDeviceGetPcieThroughput(tester, NVML_PCIE_UTIL_RX_BYTES, &b);
        //printf("TX: %f GBPS, RX: %f GBPS\n", (float)a / (1024.0f * 1024.0f), (float)b / (1024.0f *   1024.0f));
        fprintf(fptr, "%f\t%f\n", (float)a / (1024.0f * 1024.0f), (float)b / (1024.0f *   1024.0f));
        int milisec = 1; // length of time to sleep, in miliseconds
        struct timespec req = {0};
        req.tv_sec = 0;
        req.tv_nsec = milisec * 1000000L;
        nanosleep(&req, (struct timespec *)NULL);
    }
    fclose(fptr);
    nvmlShutdown();
}

//static void wa_stat(const char *s = "wa_stats.csv") {
//    uint32_t pid = fork(); 
//    if (pid == 0) {
//        child_pid = getpid();
//        const char *args[] = {"/home/shweta/pcm/pcm-pcie.x", "1", "-csv=wa.log", NULL};
//        int err = execv(args[0], (char **)args);
//        printf("ERROR IN WA: %d %d\n", err, errno);
//        fflush(stdout);
//    } else {
//    	child_pid = pid;
//    }
//	/*else {
//        while (!FINISH_WA_MONITOR);
//        kill(child_pid, SIGKILL); 
//    }*/
//}


#define START_BW_MONITOR2(file_name) \
    FINISH_BW_MONITOR = false;\
    BW_MONITOR_THREAD = std::thread(bw_stat, file_name);\

//#define START_WA_MONITOR(file_name) \
//	wa_stat(file_name);
    //FINISH_WA_MONITOR = false;\
    //WA_MONITOR_THREAD = std::thread(wa_stat, file_name);\

#define START_BW_MONITOR \
    FINISH_BW_MONITOR = false;\
    BW_MONITOR_THREAD = std::thread(bw_stat);\

#define STOP_BW_MONITOR \
    cudaDeviceSynchronize();\
    FINISH_BW_MONITOR = true;\
    BW_MONITOR_THREAD.join();\
    //BW_MONITOR_THREAD_2.join();

//#define STOP_WA_MONITOR \
//    kill(child_pid, SIGKILL);
/*    cudaDeviceSynchronize();\
    FINISH_WA_MONITOR = true;\
    printf("Hello123\n");\
    WA_MONITOR_THREAD.join();\
    printf("Hello\n");
    //WA_MONITOR_THREAD.join();\
    //std::thread(wa_end); \
*/
#else
#define START_BW_MONITOR
#define STOP_BW_MONITOR
#endif


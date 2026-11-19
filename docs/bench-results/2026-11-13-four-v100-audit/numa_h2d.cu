#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
static void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);} }
static void bind_cpu(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s); if(sched_setaffinity(0,sizeof(s),&s)){std::perror("affinity");std::exit(2);} }
static std::string placement(void* p) { std::ifstream in("/proc/self/numa_maps"); std::string l; const auto a=(unsigned long long)p; while(std::getline(in,l)){ unsigned long long b=0; if(std::sscanf(l.c_str(),"%llx",&b)==1&&a==b)return l; }return "mapping not found"; }
int main(){
 const size_t bytes=128ULL<<20; const int reps=12; int ng=0; ck(cudaGetDeviceCount(&ng));
 std::printf("{\"buffer_bytes\":%zu,\"copies_per_sample\":%d,\"samples_per_pair\":3,\"rows\":[\n",bytes,reps);
 bool first=true;
 // A separately mmapped, first-touched mapping per socket avoids heap-page reuse.
 for(int host=0;host<2;++host){
  bind_cpu(host?14:0);
  void* h=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(h==MAP_FAILED){std::perror("mmap");return 2;}
  madvise(h,bytes,MADV_NOHUGEPAGE);
  std::memset(h,host+1,bytes);
  ck(cudaHostRegister(h,bytes,cudaHostRegisterPortable));
  const auto loc=placement(h);
  for(int dev=0;dev<ng;++dev){
   ck(cudaSetDevice(dev)); void* d=nullptr;ck(cudaMalloc(&d,bytes));
   cudaStream_t s;cudaEvent_t a,b;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
   for(int i=0;i<3;++i)ck(cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice,s)); ck(cudaStreamSynchronize(s));
   double speed[3];
   for(int j=0;j<3;++j){ck(cudaEventRecord(a,s));for(int i=0;i<reps;++i)ck(cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice,s));ck(cudaEventRecord(b,s));ck(cudaEventSynchronize(b));float ms=0;ck(cudaEventElapsedTime(&ms,a,b));speed[j]=bytes*double(reps)/(ms*1e6);}
   // Confirm that the last copy reached the destination.
   unsigned char v=0;ck(cudaMemcpy(&v,d,1,cudaMemcpyDeviceToHost));if(v!=host+1)return 3;
   auto sorted=std::vector<double>(speed,speed+3);std::sort(sorted.begin(),sorted.end());
   if(!first)std::printf(",\n");first=false;
   std::printf("{\"first_touch_cpu\":%d,\"expected_numa_node\":%d,\"gpu\":%d,\"GBps\":[%.3f,%.3f,%.3f],\"median_GBps\":%.3f,\"numa_maps\":\"%s\"}",host?14:0,host,dev,speed[0],speed[1],speed[2],sorted[1],loc.c_str());std::fflush(stdout);
   ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaStreamDestroy(s));ck(cudaFree(d));
  }
  ck(cudaHostUnregister(h));munmap(h,bytes);
 }
 std::printf("\n]}\n");
}

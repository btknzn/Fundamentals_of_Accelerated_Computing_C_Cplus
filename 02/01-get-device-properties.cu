#include <stdio.h>

int main()
{
  /*
   * Assign values to these variables so that the output string below prints the
   * requested properties of the currently active GPU.
   */

  int deviceId;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int warpSize;

  cudaGetDevice(&deviceId);
  
  cudaDeviceProp props;  
  cudaGetDeviceProperties(&props, deviceId);
  multiProcessorCount = props.multiProcessorCount;
  computeCapabilityMajor = props.major;
  computeCapabilityMinor = props.minor;
  // standard solution: warpSize = props.warpSize;
  warpSize = props.maxThreadsPerBlock/32;
  
  /*
   * There should be no need to modify the output string below.
   */
  printf("Device ID: %d\nNumber of SMs: %d\n", deviceId, multiProcessorCount);
  printf("Compute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", computeCapabilityMajor, computeCapabilityMinor, warpSize);
}



#include <iostream>
#include <vectorpu.h>

using namespace std;


#define foo_cpu_flow (R)(R)(NA)
void foo_cpu(int const *x, int const *y, size_t size){
	for(int i=0; i<size; ++i){
		cout<<x[i]<<", ";
	}
	cout<<endl;
	for(int i=0; i<size; ++i){
		cout<<y[i]<<", ";
	}
	cout<<endl;
}

#define foo_cpu_overloaded (R)(NA)
void foo_cpu(int const *x, size_t size){
	for(int i=0; i<size; ++i){
		cout<<x[i]<<", ";
	}
	cout<<endl;
}

#define foo_gpu_flow (GW)(GW)(NA)
__global__
void foo_gpu( int *x, int *y, size_t size){
	for(int i=0; i<size; ++i){
		x[i]=101;
		y[i]=202;
	}
}

int main()
{

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;




	// Reset the device and exit
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	vectorpu::vector<int> x(10), y(10);

	CALL( (foo_gpu) ((<<<1,1>>>)) ((x, y, 10)) );

	CALL( (foo_cpu) (()) ((x, y, 10)) );

	CALLC( (foo_cpu) (()) ((x, 10)) (foo_cpu_overloaded) );


	return EXIT_SUCCESS;
}

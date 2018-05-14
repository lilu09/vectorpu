
#include <iostream>
#include <vectorpu.h>

using namespace std;

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

	vectorpu::vector<int> x(10,1), y(10,2);

	std::copy(RI(y), REI(y), std::ostream_iterator<int>(std::cout, ", "));
	cout<<(char)8<<(char)8<<' '<<endl;

	vectorpu::copy<int>(RI(x), REI(x), GWI(y) );


	std::copy(RI(y), REI(y), std::ostream_iterator<int>(std::cout, ", "));
	cout<<(char)8<<(char)8<<' '<<endl;

	return EXIT_SUCCESS;
}

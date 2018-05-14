
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

	vectorpu::vector<float> x{1.2,1.3,2,3};

	std::copy(RI(x), REI(x), std::ostream_iterator<float>(std::cout, ", "));
	cout<<(char)8<<(char)8<<' '<<endl;


	int i=0;
	std::generate(WI(x), WEI(x), [&i]{return i++;} );

	std::copy(RI(x), REI(x), std::ostream_iterator<float>(std::cout, ", "));
	cout<<(char)8<<(char)8<<' '<<endl;

	
	




	return EXIT_SUCCESS;
}

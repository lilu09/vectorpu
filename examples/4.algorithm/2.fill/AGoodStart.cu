
#include <iostream>
#include <iterator>
#include <vectorpu.h>

using namespace std;




int main()
{

	cudaError_t err = cudaSuccess;

	err = cudaDeviceReset();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	const std::size_t N=3;

	vectorpu::vector<int> x(N);

	//write on gpu side
	vectorpu::fill(GWI(x),GWEI(x),199);

	//read on cpu side
	std::copy(RI(x), REI(x), std::ostream_iterator<int>(std::cout, " "));
	std::cout<<std::endl;



	return EXIT_SUCCESS;
}

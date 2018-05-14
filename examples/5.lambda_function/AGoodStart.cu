
#include <iostream>
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


	vectorpu::vector<int> z(3,0);

/*------------------------------------------------------------------------------------*/

#define lambda(z) \
			for(std::size_t i=0; i<N; ++i)  \
			{ \
				z[i]=3; \
			}

#define VECTORPU_DESCRIBE VECTORPU_META_DATA(int, z, W)
VECTORPU_LAMBDA_GEN
/*------------------------------------------------------------------------------------*/

	std::copy(RI(z), REI(z), std::ostream_iterator<float>(std::cout, ", "));
	cout<<(char)8<<(char)8<<' '<<endl;



	return EXIT_SUCCESS;
}

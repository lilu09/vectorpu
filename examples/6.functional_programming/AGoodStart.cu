
/*#include <iostream>*/
#include <stdio.h>
#include <iterator>
#include <vectorpu.h>

#include <thrust/sort.h>
#include <algorithm>
#include <string>

using namespace std;


struct my_set{
	template <class T>
	__host__ __device__
		void operator() (T &x) {  // function:
			x+=1;
		}
};


struct my_print{
	template <class T>
	__host__ __device__
		void operator() (T const &x) {  // function:
			/*std::cout << ' ' << i;*/
			printf("%d,",x);
		}
};

struct my_sum{
	my_sum():sum(0){}
	int sum;
	template <class T>
	__host__ __device__
		void operator() (T const &x) {  // function:
			sum+=x;
		}
};




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


	vectorpu::for_each<int>(GWI(x), GWEI(x), [] __device__ (int &x){x=11011;} );

	vectorpu::for_each<int>(GRWI(x), GRWEI(x), my_set() );

	vectorpu::for_each<int>(RI(x), WEI(x), [](int const x) {cout<<x<<",";} );
	cout<<(char)8<<' '<<endl;


	vector<int> correct_result{11012,11012,11012};
	assert( equal( x.begin(), x.end(), correct_result.begin() ) );
	
	




	return EXIT_SUCCESS;
}

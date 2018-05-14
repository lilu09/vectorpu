
#include <iostream>
#include <iterator>
#include <vectorpu.h>

#include <thrust/sort.h>
#include <algorithm>
#include <string>

using namespace std;

int RandomNumber () { return (std::rand()%100); }

#define FUNC_MULTIPLE_TIME_USE_1_META (GRW)(NA)
__global__
void func_multiple_time_use_1(int *x, std::size_t N)
{
	for(std::size_t i=0; i<N; ++i)
	{
		x[i]+=4;
	}
	
}

struct My_Type{
	int x;
	int y;
	void operator=(int val)
	{
		x=val;
		y=val;
	}
};

std::ostream& operator<<(std::ostream& os, const My_Type& obj){
	os<<obj.x<<" ";
	os<<obj.y<<std::endl;
	return os;
}

__host__ __device__
bool operator<(const My_Type& o1, const My_Type& o2){
	return o1.x<o2.x;
}




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

	vectorpu::vector<My_Type> x(N);

	cout<<"Init"<<endl;
	cout<<string(50,'-')<<endl;
	std::copy(RI(x), RI(x)+x.size(), std::ostream_iterator<My_Type>(std::cout, ""));
	cout<<string(50,'-')<<endl<<endl;

	for(int i=0;i<x.size(); ++i){
		(*WI(x[i])) = i;
	}

	cout<<"CPU Read"<<endl;
	cout<<string(50,'-')<<endl;
	std::copy(RI(x), RI(x)+x.size(), std::ostream_iterator<My_Type>(std::cout, ""));
	cout<<string(50,'-')<<endl<<endl;


	std::generate(WI(x), WI(x)+x.size(), RandomNumber);

	cout<<"CPU sequence it"<<endl;
	cout<<string(50,'-')<<endl;
	std::copy(RI(x), RI(x)+x.size(), std::ostream_iterator<My_Type>(std::cout, ""));
	cout<<string(50,'-')<<endl<<endl;


	thrust::sort(GRWI(x), GRWI(x)+x.size());


	cout<<"GPU sorted back"<<endl;
	cout<<string(50,'-')<<endl;
	std::copy(RI(x), RI(x)+x.size(), std::ostream_iterator<My_Type>(std::cout, ""));
	cout<<string(50,'-')<<endl<<endl;





	return EXIT_SUCCESS;
}

#include <bits/stdc++.h>
#include <vectorpu.h>

using namespace std;

namespace vectorpu{


	template <class T>
	__global__
		void reduce(T *p, size_t size, size_t step){
			size_t i = blockIdx.x*blockDim.x+threadIdx.x;
			size_t current_size=size;

			for(int j=0; j<step; ++j){
				current_size=current_size/2;
				if (i < current_size)
					p[i] += p[i+current_size];
			}
		}

}


int main(void)
{

	vectorpu::vector<int> x(8);

	int n=1;
	generate(WI(x), WEI(x), [&n]{return n++;});

	copy(RI(x), REI(x), ostream_iterator<int>(cout, " ") );
	cout<<endl;

	vectorpu::reduce<<<1,8>>>( GRW(x), x.array_size, log2(x.array_size) );

	//declare the parco_vector later
	//will allow parco_vector inherit x's coherent state
	//thus no SR needed
	vectorpu::parco_vector<int> x1(x, x.begin(), x.begin()+1);

	cout<<*R(x1)<<endl;

	auto result=*R(x1);

	assert(result == 36);



}

#include <bits/stdc++.h>
#include <assert.h>
#include <vectorpu.h>

using namespace std;


namespace vectorpu{

	template <class T>
	__global__
		void hillis_steele_inclusive_scan(T *p, size_t const step, size_t size){

			T i=blockIdx.x*blockDim.x+threadIdx.x;
			T start_id;

			for(size_t j=1;j<=step;++j){
				start_id = powf(2,j-1);


				if( i>=start_id ){
					p[i] += p[ i - start_id ];
				}
			}
		}


	template <class T>
	__host__ __device__
		void downsweep(T *p, size_t move_id, size_t reduce_id){
			T temp = p[move_id];
			p[move_id] = p[reduce_id];
			p[reduce_id] += temp;
		}

	template <class T>
	__global__
		void blelloch_exclusive_scan(T *p, size_t const step, size_t size){

			size_t i=blockIdx.x*blockDim.x+threadIdx.x;
			size_t hop, mask_id, expected_mask_id;

			for(size_t j=1;j<=step;++j){
				hop=powf(2,j-1); //1,2,4 ...
				mask_id = i % (2*hop); // i%2=1, i%4=3, i%8=7
				expected_mask_id=powf(2,j)-1;  //1,3,7 ...
				if( mask_id == expected_mask_id ){
					p[i] += p[ i - hop ];
				}
			}

			if ( i == (size - 1) )
				p[i] = 0;

			//downsweep
			for(size_t j=step;j>=1;--j){
				hop=powf(2,j-1); //1,2,4 ...
				mask_id = i % (2*hop); // i%2=1, i%4=3, i%8=7
				expected_mask_id=powf(2,j)-1;  //1,3,7 ...
				if( mask_id == expected_mask_id ){
					downsweep(p, i-hop, i);
				}
			}
		
		}

}


int main(void)
{

	vectorpu::vector<int> x(8);

	int n=1;
	generate(WI(x), WEI(x), [&n]{return n++;} );

	copy(RI(x), REI(x), ostream_iterator<int>(cout, ",") );
	cout<<(char)8<<' '<<endl;

	size_t step=log2( x.size() );

	vectorpu::blelloch_exclusive_scan<<<1,x.size()>>>(GRW(x), step, x.size() );
	/*vectorpu::hillis_steele_inclusive_scan<<<1,x.size()>>>(GRW(x), step, x.size() );*/

	copy(RI(x), REI(x), ostream_iterator<int>(cout, ",") );
	cout<<(char)8<<' '<<endl;


	std::vector<int> correct_result{0,1,3,6,10,15,21,28};

	assert( equal(x.cbegin(), x.cend(), correct_result.cbegin() ) );



}

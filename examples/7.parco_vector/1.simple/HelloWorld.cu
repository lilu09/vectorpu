#include <bits/stdc++.h>
#include <vectorpu.h>

using namespace std;
//using std::cout;
//using std::endl;
//using std::vector;
//using std::string;


struct my_set{
	template <class T>
	__host__ __device__
		void operator() (T &x) {  // function:
			x=101;
		}
};

struct my_set2{
	template <class T>
	__host__ __device__
		void operator() (T &x) {  // function:
			x=102;
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

int main()
{

	vectorpu::vector<int> x(10);


	vectorpu::parco_vector<int> y(x, x.begin(), x.begin()+2); //here you can also let x know y is count on himself.

	//do operations on parco_vector
	vectorpu::for_each<int>(GWI(y), GWEI(y), my_set() );
	vectorpu::for_each<int>(GWI(y), GWEI(y), my_set2() );
	/*vectorpu::for_each<float>(RI(y), 2, my_print() ); //if with this line, all states good, no more efforts need*/
	               					  //otherwise, you have a state with GPU valid, but CPU invalid*/
	
	SR(y);  //Sync for Read on y: we rely on the user to sync, instead of doing a lot of checks in which most are unnecessary 
		//in the future version of VectorPU this is like to be unnecessary.





	vectorpu::for_each<int>(RI(x), REI(x), my_print() );  //what is necessary is only to transfer part of x, which is y to CPU memory from GPU memory
	cout<<(char)8<<' '<<endl;			   //now x need to check y's coherent flag as well
							   //but this check can not be avoided if no parco_vector on x, increase unnecessary overhead!
	

	vector<int> correct_result{102,102,0,0,0,0,0,0,0,0};
	assert( equal( x.begin(), x.end(), correct_result.begin() ) );


	return 0;
}

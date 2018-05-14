

#include <algorithm>
#include <iterator>
#include <vector>


#if defined(XPDL_NUM_OF_GPUS) && (XPDL_NUM_OF_GPUS>=1)
//#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif



namespace vectorpu{



#if defined(XPDL_NUM_OF_GPUS) && (XPDL_NUM_OF_GPUS>=1)

//base vector, only contain gpu vector, coherent flag, and coherent algorithm
//only for reuse, perferably no polymophism


//minimum overhead vector: keep the middleware layer as thin as possible on one GPU
/*{{{*/
template <class T, class Index_Type=std::size_t>
struct min_vector : public std::vector<T>, public thrust::device_vector<T> {
/*{{{*/
        explicit min_vector(Index_Type _array_size):
       	 std::vector<T>::vector(_array_size),
       	 thrust::device_vector<T>::device_vector(_array_size),
	 cpu_coherent_unit(true),
	 gpu_coherent_unit(true),
       	 array_size(_array_size),
	 start_pos(0) {}

        explicit min_vector(Index_Type _array_size, T const &val):
       	 std::vector<T>::vector(_array_size, val),
       	 thrust::device_vector<T>::device_vector(_array_size, val),
	 cpu_coherent_unit(true),
	 gpu_coherent_unit(true),
       	 array_size(_array_size),
	 start_pos(0) {}

	//allow constructor by iterator, initialize cpu and gpu memory by copy
	template <class InputIterator>
	explicit min_vector(InputIterator start_iter, InputIterator end_iter):
			std::vector<T>::vector(start_iter, end_iter),
		thrust::device_vector<T>::device_vector(start_iter, end_iter),
		cpu_coherent_unit(true),
		gpu_coherent_unit(true),
		array_size(end_iter-start_iter), 
		 start_pos(0) {}

	explicit min_vector(std::initializer_list<T> const &l):
	std::vector<T>::vector(l),
       	thrust::device_vector<T>::device_vector(std::vector<T>::begin(), std::vector<T>::end() ),
	cpu_coherent_unit(true),
	gpu_coherent_unit(true),
	array_size(l.size()),
	start_pos(0)
	{}


	Index_Type size(){ return array_size; }

        const T& r(const Index_Type& _array_index) { return *(std::vector<T>::begin()+_array_index); }

        void w(const Index_Type& _array_index, const T& new_val) { *(std::vector<T>::begin()+_array_index)=new_val; }

        const T gr(const Index_Type& _array_index) { T const t=static_cast<thrust::device_vector<T> >(*this)[_array_index]; return t; }

        void gw(const Index_Type& _array_index, const T& new_val) { *(thrust::device_vector<T>::begin()+_array_index)=new_val; }

        bool cpu_coherent_unit;
        bool gpu_coherent_unit;
	Index_Type array_size;
	Index_Type start_pos;


	void show_cpu_memory(){
		std::copy(begin(), end(), std::ostream_iterator<T>(std::cout, " ") );
		std::cout<<std::endl;
	}

	//---------------------------------Iterator-------------------------------------//
	typename std::vector<T>::iterator begin(){
		return std::vector<T>::begin();
	}
	typename std::vector<T>::const_iterator cbegin(){
		return std::vector<T>::cbegin();
	}

	typename std::vector<T>::iterator end(){
		return std::vector<T>::end();
	}
	typename std::vector<T>::const_iterator cend(){
		return std::vector<T>::cend();
	}

	typename thrust::device_vector<T>::iterator gbegin(){
		return thrust::device_vector<T>::begin();
	}
	typename thrust::device_vector<T>::const_iterator gcbegin(){
		return thrust::device_vector<T>::cbegin();
	}
	typename thrust::device_vector<T>::iterator gend(){
		return thrust::device_vector<T>::end();
	}


	typename std::vector<T>::const_iterator get_cpu_r_begin(){
		Index_Type _start_pos=start_pos;
		start_pos=0;
	 	coherent_on_cpu_r();
		return std::vector<T>::begin()+_start_pos;
	}
	typename std::vector<T>::const_iterator get_cpu_r_end(){
		return std::vector<T>::end();
	}

	typename std::vector<T>::iterator get_cpu_w_begin(){
		Index_Type _start_pos=start_pos;
		start_pos=0;
	 	coherent_on_cpu_w();
		return std::vector<T>::begin()+_start_pos;
	}
	typename std::vector<T>::iterator get_cpu_w_end(){
		return std::vector<T>::end();
	}

	typename std::vector<T>::iterator get_cpu_rw_begin(){
		Index_Type _start_pos=start_pos;
		start_pos=0;
	 	coherent_on_cpu_rw();
		return std::vector<T>::begin()+_start_pos;
	}
	typename std::vector<T>::iterator get_cpu_rw_end(){
		return std::vector<T>::end();
	}

	typename thrust::device_vector<T>::const_iterator get_gpu_r_begin(){
		Index_Type _start_pos=start_pos;
		start_pos=0;
	 	coherent_on_gpu_r();
		return thrust::device_vector<T>::begin()+_start_pos;
	}
	typename thrust::device_vector<T>::const_iterator get_gpu_r_end(){
		return thrust::device_vector<T>::end();
	}

	typename thrust::device_vector<T>::iterator get_gpu_w_begin(){
		Index_Type _start_pos=start_pos;
		start_pos=0;
	 	coherent_on_gpu_w();
		return thrust::device_vector<T>::begin()+_start_pos;
	}
	typename thrust::device_vector<T>::iterator get_gpu_w_end(){
		return thrust::device_vector<T>::end();
	}

	typename thrust::device_vector<T>::iterator get_gpu_rw_begin(){
		Index_Type _start_pos=start_pos;
		start_pos=0;
	 	coherent_on_gpu_rw();
		return thrust::device_vector<T>::begin()+_start_pos;
	}
	typename thrust::device_vector<T>::iterator get_gpu_rw_end(){
		return thrust::device_vector<T>::end();
	}
	//---------------------------------Iterator End---------------------------------//

	//typename std::vector<T>::iterator begin(){
		//return std::vector<T>::begin()+start_pos;
	//}
	//typename thrust::device_vector<T>::iterator gtbegin(){
		//return thrust::device_vector<T>::begin()+start_pos;
	//}


        void reset(){
       	 cpu_coherent_unit=true;
       	 gpu_coherent_unit=true;
       	 std::fill(std::vector<T>::begin(), std::vector<T>::end(), 0);
       	 upload();
        }

	min_vector& operator[] (Index_Type _start_pos) { start_pos=_start_pos;return *this;}

        void upload() {
       	 thrust::copy(
       			 std::vector<T>::begin(),
       			 std::vector<T>::end(),
       			 thrust::device_vector<T>::begin()
       		     );

        }
        void download() {
       	 thrust::copy(
       			 thrust::device_vector<T>::begin(),
       			 thrust::device_vector<T>::end(),
       			 std::vector<T>::begin()
       		     );

        }

        T* cpu(){
       	 return std::vector<T>::data();
        }


        T* gpu(){
       	 return thrust::raw_pointer_cast(& (* thrust::device_vector<T>::begin() ) );
        }

	//for TFO
        void download(Index_Type _start, Index_Type _end) {
       	 thrust::copy(
       			 thrust::device_vector<T>::begin()+_start,
       			 thrust::device_vector<T>::begin()+_end,
       			 std::vector<T>::begin()+_start
       		     );

        }
	//only transfer the first few elements, good for reduce where at the end only the first one is needed
        void download(Index_Type _size) {
       	 thrust::copy(
       			 thrust::device_vector<T>::begin(),
       			 thrust::device_vector<T>::begin()+_size,
       			 std::vector<T>::begin()
       		     );

        }

	T const *get_cpu_r_raw_ptr_no_management(Index_Type _size){
		download(_size);
		return std::vector<T>::data();
	}


	//--------------------------------Lazy Coherence--------------------------------//
	void coherent_on_cpu_r(){
       	 if( !cpu_coherent_unit ){
       		 download();
       		 cpu_coherent_unit=true;
       	 }
	}
	void coherent_on_cpu_w(){
       	 cpu_coherent_unit=true;
       	 gpu_coherent_unit=false;
	}
	void coherent_on_cpu_rw(){
       	 if( !cpu_coherent_unit ){
       		 download();
       		 cpu_coherent_unit=true;
       	 }
       	 gpu_coherent_unit=false;
	}
	void coherent_on_gpu_r(){
       	 if( !gpu_coherent_unit ){
       		 upload();
       		 gpu_coherent_unit=true;
       	 }
	}
	void coherent_on_gpu_w(){
       	 gpu_coherent_unit=true;
       	 cpu_coherent_unit=false;
	}
	void coherent_on_gpu_rw(){
       	 if( !gpu_coherent_unit ){
       		 upload();
       		 gpu_coherent_unit=true;
       	 }
       	 cpu_coherent_unit=false;
	}

	//--------------------------------Lazy Coherence End----------------------------//


	//---------------------------------Raw Pointer----------------------------------//

        const T* get_cpu_r_raw_ptr(){
                //if( !cpu_coherent_unit ){
                        //download();
                        //cpu_coherent_unit=true;
                //}
	 coherent_on_cpu_r();
       	 return this->std::vector<T>::data();
        }

        T* get_cpu_w_raw_ptr(){
                //cpu_coherent_unit=true;
                //gpu_coherent_unit=false;
	 coherent_on_cpu_w();
       	 return this->std::vector<T>::data();
        }

        T* get_cpu_rw_raw_ptr(){
                //if( !cpu_coherent_unit ){
                        //download();
                        //cpu_coherent_unit=true;
                //}
                //gpu_coherent_unit=false;
	 coherent_on_cpu_rw();
       	 return this->std::vector<T>::data();
        }

        const T* get_gpu_r_raw_ptr(){
                //if( !gpu_coherent_unit ){
                        //upload();
                        //gpu_coherent_unit=true;
                //}
	 coherent_on_gpu_r();
       	 return thrust::raw_pointer_cast(& (* thrust::device_vector<T>::begin() ) );
        }

        T* get_gpu_w_raw_ptr(){
                //gpu_coherent_unit=true;
                //cpu_coherent_unit=false;
	 coherent_on_gpu_w();
       	 return thrust::raw_pointer_cast(& (* thrust::device_vector<T>::begin() ) );

        }

        T* get_gpu_rw_raw_ptr(){
                //if( !gpu_coherent_unit ){
                        //upload();
                        //gpu_coherent_unit=true;
                //}
                //cpu_coherent_unit=false;
	 coherent_on_gpu_rw();
       	 return thrust::raw_pointer_cast(& (* thrust::device_vector<T>::begin() ) );
        }
	//---------------------------------Raw Pointer End------------------------------//
	/*}}}*/
};
/*}}}*/

//partial coherence vector:
/*{{{*/
template <class T, class Index_Type=std::size_t>
struct parco_vector {
/*{{{*/
	using cpu_iter_type=typename std::vector<T>::iterator;
	using gpu_iter_type=typename thrust::device_vector<T>::iterator;

	//it is hard to add const to _host_vector
	//also hard to use gcbegin, may be this is the reason for the first hard
	explicit parco_vector(min_vector<T> &_host_vector,  cpu_iter_type const _start_iter, cpu_iter_type const _end_iter):
		cpu_start_iter(_start_iter), cpu_end_iter(_end_iter),
		gpu_start_iter(_host_vector.gbegin()+(_start_iter-_host_vector.cbegin() ) ), gpu_end_iter(_host_vector.gbegin()+ (_end_iter-_host_vector.cbegin()) ),
		array_size(_end_iter-_start_iter),
		cpu_coherent_unit(_host_vector.cpu_coherent_unit),
		gpu_coherent_unit(_host_vector.gpu_coherent_unit)
	{}

	cpu_iter_type const cpu_start_iter;
	cpu_iter_type const cpu_end_iter;
	gpu_iter_type gpu_start_iter;
	gpu_iter_type gpu_end_iter;
	Index_Type array_size;
        bool cpu_coherent_unit;
        bool gpu_coherent_unit;

	auto begin() -> decltype(cpu_start_iter) {return cpu_start_iter;}
	auto end() -> decltype(cpu_end_iter) {return cpu_end_iter;}
	auto gbegin() -> decltype(gpu_start_iter) {return gpu_start_iter;}
	auto gend() -> decltype(gpu_end_iter) {return gpu_end_iter;}

	void show_cpu_memory(){
		std::copy(cpu_start_iter, cpu_end_iter, std::ostream_iterator<T>(std::cout, " ") );
		std::cout<<std::endl;
	}

	//-----------------------------------------------------------------------------//
        void upload() {
       	 thrust::copy(
       			 cpu_start_iter,
       			 cpu_end_iter,
       			 gpu_start_iter
       		     );
        }
        void download() {
       	 thrust::copy(
       			 gpu_start_iter,
       			 gpu_end_iter,
       			 cpu_start_iter
       		     );
        }


	//-----------------------------------------------------------------------------//

	typename std::vector<T>::const_iterator get_cpu_r_begin(){
	 	coherent_on_cpu_r();
		return cpu_start_iter;
	}
	typename std::vector<T>::const_iterator get_cpu_r_end(){
		return cpu_end_iter;
	}

	typename std::vector<T>::iterator get_cpu_w_begin(){
	 	coherent_on_cpu_w();
		return cpu_start_iter;
	}
	typename std::vector<T>::iterator get_cpu_w_end(){
		return cpu_end_iter;
	}

	typename std::vector<T>::iterator get_cpu_rw_begin(){
	 	coherent_on_cpu_rw();
		return cpu_start_iter;
	}
	typename std::vector<T>::iterator get_cpu_rw_end(){
		return cpu_end_iter;
	}

	typename thrust::device_vector<T>::const_iterator get_gpu_r_begin(){
	 	coherent_on_gpu_r();
		return gpu_start_iter;
	}
	typename thrust::device_vector<T>::const_iterator get_gpu_r_end(){
		return gpu_end_iter;
	}

	typename thrust::device_vector<T>::iterator get_gpu_w_begin(){
	 	coherent_on_gpu_w();
		return gpu_start_iter;
	}
	typename thrust::device_vector<T>::iterator get_gpu_w_end(){
		return gpu_end_iter;
	}

	typename thrust::device_vector<T>::iterator get_gpu_rw_begin(){
	 	coherent_on_gpu_rw();
		return gpu_start_iter;
	}
	typename thrust::device_vector<T>::iterator get_gpu_rw_end(){
		return gpu_end_iter;
	}

	//-----------------------------------------------------------------------------//
	void coherent_on_cpu_r(){
       	 if( !cpu_coherent_unit ){
       		 download();
       		 cpu_coherent_unit=true;
       	 }
	}
	void coherent_on_cpu_w(){
       	 cpu_coherent_unit=true;
       	 gpu_coherent_unit=false;
	}
	void coherent_on_cpu_rw(){
       	 if( !cpu_coherent_unit ){
       		 download();
       		 cpu_coherent_unit=true;
       	 }
       	 gpu_coherent_unit=false;
	}
	void coherent_on_gpu_r(){
       	 if( !gpu_coherent_unit ){
       		 upload();
       		 gpu_coherent_unit=true;
       	 }
	}
	void coherent_on_gpu_w(){
       	 gpu_coherent_unit=true;
       	 cpu_coherent_unit=false;
	}
	void coherent_on_gpu_rw(){
       	 if( !gpu_coherent_unit ){
       		 upload();
       		 gpu_coherent_unit=true;
       	 }
       	 cpu_coherent_unit=false;
	}
	//-----------------------------------------------------------------------------//
        const T* get_cpu_r_raw_ptr(){
	 coherent_on_cpu_r();
       	 return &(*cpu_start_iter) ;
        }

        T* get_cpu_w_raw_ptr(){
	 coherent_on_cpu_w();
       	 return &(*cpu_start_iter);
        }

        T* get_cpu_rw_raw_ptr(){
	 coherent_on_cpu_rw();
       	 return &(*cpu_start_iter);
        }

        const T* get_gpu_r_raw_ptr(){
	 coherent_on_gpu_r();
       	 return thrust::raw_pointer_cast(& (* gpu_start_iter ) );
        }

        T* get_gpu_w_raw_ptr(){
	 coherent_on_gpu_w();
       	 return thrust::raw_pointer_cast(& (* gpu_start_iter ) );

        }

        T* get_gpu_rw_raw_ptr(){
	 coherent_on_gpu_rw();
       	 return thrust::raw_pointer_cast(& (* gpu_start_iter ) );
        }
	//-----------------------------------------------------------------------------//
/*}}}*/
};
/*}}}*/

//multi-gpu vector: use several minimum overhead vector
/*{{{*/

#ifdef XPDL_NUM_OF_GPUS

#if XPDL_NUM_OF_GPUS > 1  

#undef BOOST_PP_VARIADICS
#define BOOST_PP_VARIADICS 1
#include <boost/preprocessor.hpp>


//vectorpu big_vector macro library
/*{{{*/

//zip_pointer
/*{{{*/
#define VECTORPU_EACH_POINTER(z, n, text) T *p ## n; Index_Type size ## n;


#define VECTORPU_ZIP_POINTER(_GPU_NUM) \
template <class T, typename Index_Type=size_t> \
struct zip_pointer { \
	BOOST_PP_REPEAT(_GPU_NUM, VECTORPU_EACH_POINTER, ) \
	Index_Type size; \
};
/*}}}*/

//boundary calculation
/*{{{*/
#define VECTORPU_EACH_BOUNDARY_LIST_ASSIGNMENT(z,n,text) boundary ## n (_array_size*BOOST_PP_INC(n)/text ) BOOST_PP_COMMA_IF( BOOST_PP_NOT_EQUAL( n, BOOST_PP_SUB(text,2)))
#define VECTORPU_ALL_BOUNDARIES_LIST_ASSIGNMENT(_GPU_NUM) BOOST_PP_REPEAT(BOOST_PP_DEC(_GPU_NUM), VECTORPU_EACH_BOUNDARY_LIST_ASSIGNMENT, _GPU_NUM)

#define VECTORPU_EACH_BOUNDARY_ASSIGNMENT(z,n,text) boundary ## n =_array_size*BOOST_PP_INC(n)/text;
#define VECTORPU_ALL_BOUNDARIES_ASSIGNMENT(_GPU_NUM) BOOST_PP_REPEAT(BOOST_PP_DEC(_GPU_NUM), VECTORPU_EACH_BOUNDARY_ASSIGNMENT, _GPU_NUM)

//VECTORPU_ALL_BOUNDARIES_ASSIGNMENT(XPDL_NUM_OF_GPUS)
//VECTORPU_ALL_BOUNDARIES_LIST_ASSIGNMENT(XPDL_NUM_OF_GPUS)
/*}}}*/

//boundary declaration
/*{{{*/
#define VECTORPU_EACH_BOUNDARY_DECL(z,n,text) Index_Type boundary ## n;
#define VECTORPU_ALL_BOUNDARIES_DECL(_GPU_NUM) BOOST_PP_REPEAT(BOOST_PP_DEC(_GPU_NUM), VECTORPU_EACH_BOUNDARY_DECL, )

//VECTORPU_ALL_BOUNDARIES_DECL(XPDL_NUM_OF_GPUS)
/*}}}*/

//ghost_vector initialization
/*{{{*/

#define START_ITER(n) std::vector<T>::begin() BOOST_PP_IF( BOOST_PP_EQUAL(n,0), , +BOOST_PP_CAT( boundary, BOOST_PP_DEC(n) )   )
#define END_ITER(n, max) BOOST_PP_IF( BOOST_PP_EQUAL(n, BOOST_PP_DEC(max) ), std::vector<T>::end(), BOOST_PP_CAT(std::vector<T>::begin()+boundary, n)  )


#define VECTORPU_EACH_GHOST_VECTOR_ASSIGNMENT(z,n,text) ghost ## n= new ghost_vector<T, n > ( START_ITER(n), END_ITER(n, text) );
#define VECTORPU_ALL_GHOST_VECTOR_ASSIGNMENT(_GPU_NUM) BOOST_PP_REPEAT(_GPU_NUM, VECTORPU_EACH_GHOST_VECTOR_ASSIGNMENT, _GPU_NUM )

//VECTORPU_ALL_GHOST_VECTOR_ASSIGNMENT(XPDL_NUM_OF_GPUS)

#define VECTORPU_EACH_GHOST_VECTOR_LIST_ASSIGNMENT(z,n,text) ghost ## n( new ghost_vector<T, n > ( START_ITER(n), END_ITER(n, text) ) ) BOOST_PP_COMMA_IF( BOOST_PP_NOT_EQUAL( n, BOOST_PP_SUB(text,1)))
#define VECTORPU_ALL_GHOST_VECTOR_LIST_ASSIGNMENT(_GPU_NUM) BOOST_PP_REPEAT(_GPU_NUM, VECTORPU_EACH_GHOST_VECTOR_LIST_ASSIGNMENT, _GPU_NUM )

//VECTORPU_ALL_GHOST_VECTOR_LIST_ASSIGNMENT(XPDL_NUM_OF_GPUS)

#define VECTORPU_EACH_GHOST_VECTOR_LIST_ASSIGNMENT_WITH_VAL(z,n,text) ghost ## n( new ghost_vector<T, n > ( START_ITER(n), END_ITER(n, text), _val ) ) BOOST_PP_COMMA_IF( BOOST_PP_NOT_EQUAL( n, BOOST_PP_SUB(text,1)))
#define VECTORPU_ALL_GHOST_VECTOR_LIST_ASSIGNMENT_WITH_VAL(_GPU_NUM) BOOST_PP_REPEAT(_GPU_NUM, VECTORPU_EACH_GHOST_VECTOR_LIST_ASSIGNMENT_WITH_VAL, _GPU_NUM )

//VECTORPU_ALL_GHOST_VECTOR_LIST_ASSIGNMENT_WITH_VAL(XPDL_NUM_OF_GPUS)

/*}}}*/

//ghost_vector declaration
/*{{{*/
#define VECTORPU_EACH_GHOST_VECTOR_DECL(z,n,text) ghost_vector<T, n >* ghost ## n;
#define VECTORPU_ALL_GHOST_VECTOR_DECL(_GPU_NUM) BOOST_PP_REPEAT(_GPU_NUM, VECTORPU_EACH_GHOST_VECTOR_DECL, )

//VECTORPU_ALL_GHOST_VECTOR_DECL(XPDL_NUM_OF_GPUS)
/*}}}*/

//zip_pointer assignment
/*{{{*/

//TODO: you may merge the two cases (cpu, gpu) by pass a array instead, like (cpu)(r)
#define VECTORPU_EACH_GHOST_VECTOR_DECL_CPU(z,n,text)  ghost##n -> coherent_on_cpu_##text (); result.p##n =ghost##n ->ptr(); result.size##n =ghost##n ->array_size;
#define VECTORPU_ALL_GHOST_VECTOR_DECL_CPU(_GPU_NUM, flow_annotation) BOOST_PP_REPEAT(_GPU_NUM, VECTORPU_EACH_GHOST_VECTOR_DECL_CPU, flow_annotation)

#define VECTORPU_EACH_GHOST_VECTOR_DECL_GPU(z,n,text) cudaSetDevice(n); ghost##n -> coherent_on_cpu_##text (); result.p##n =ghost##n ->ptr(); result.size##n =ghost##n ->array_size;
#define VECTORPU_ALL_GHOST_VECTOR_DECL_GPU(_GPU_NUM, flow_annotation) BOOST_PP_REPEAT(_GPU_NUM, VECTORPU_EACH_GHOST_VECTOR_DECL_GPU, flow_annotation)

//VECTORPU_ALL_GHOST_VECTOR_DECL_CPU(XPDL_NUM_OF_GPUS, r)
//VECTORPU_ALL_GHOST_VECTOR_DECL_CPU(XPDL_NUM_OF_GPUS, w)
//VECTORPU_ALL_GHOST_VECTOR_DECL_CPU(XPDL_NUM_OF_GPUS, rw)

//VECTORPU_ALL_GHOST_VECTOR_DECL_GPU(XPDL_NUM_OF_GPUS, r)
//VECTORPU_ALL_GHOST_VECTOR_DECL_GPU(XPDL_NUM_OF_GPUS, w)
//VECTORPU_ALL_GHOST_VECTOR_DECL_GPU(XPDL_NUM_OF_GPUS, rw)
/*}}}*/

/*}}}*/

//keep the possibility for 
//both efficient sequential and parallel algorithm
//This is the return value for marked variable

//now generated instead of hard-coded
//template <class T, typename Index_Type=size_t>
//struct zip_pointer {
	//T *p0;
	//Index_Type size0;
	//T *p1;
	//Index_Type size1;
	//Index_Type size;
//};

VECTORPU_ZIP_POINTER(XPDL_NUM_OF_GPUS)

//Another kind of vector, 
//for CPU memory, it only hold two iterators
//for GPU memory, it hold a real memory
//and have the same coherence as min_vector
//in this way, we have one big cpu memory as continous,
//thus possible to perform efficient operations on them as a whole, which also allows kernel fusion
//also, we reuse the coherence mechanism


//old code, has the problem that pointer can not invoke upload
/*{{{*/
/*
template <class T, typename Index_Type=std::size_t>
struct ghost_vector : public thrust::device_vector<T> {

	using iter_type=typename std::vector<T>::iterator;

	explicit ghost_vector(iter_type const _start_iter, iter_type const _end_iter):
		start_iter(_start_iter), end_iter(_end_iter),
		array_size(_end_iter-_start_iter),
		cpu_coherent_unit(true),
		gpu_coherent_unit(true),
       	 	thrust::device_vector<T>::device_vector(array_size) {}

	iter_type const start_iter;
	iter_type const end_iter;
	Index_Type array_size;
        bool cpu_coherent_unit;
        bool gpu_coherent_unit;

	auto begin() -> decltype(start_iter) {return start_iter;}
	auto end() -> decltype(end_iter) {return end_iter;}

	T* ptr()  {return &(*start_iter);}
	T* gptr()  {return thrust::raw_pointer_cast(& (* thrust::device_vector<T>::begin() ) );}

        void upload() {
       	 thrust::copy(
       			 start_iter,
       			 end_iter,
       			 thrust::device_vector<T>::begin()
       		     );

        }
        void download() {
       	 thrust::copy(
       			 thrust::device_vector<T>::begin(),
       			 thrust::device_vector<T>::end(),
       			 start_iter
       		     );

        }

	void coherent_on_cpu_r(){
       	 if( !cpu_coherent_unit ){
       		 download();
       		 cpu_coherent_unit=true;
       	 }
	}

	void show_cpu_memory(){
		std::copy(start_iter, end_iter, std::ostream_iterator<T>(std::cout, " ") );
		std::cout<<std::endl;
	}

	//auto gbegin() 
	//auto gend() 

};

template <class T, class Index_Type=std::size_t>
struct big_vector: public std::vector<T> {
	explicit big_vector(Index_Type _array_size):
       	 std::vector<T>::vector(_array_size),
       	 array_size(_array_size) {
	 
		//can move to list initialization
		//generate the following code
		//boundaries[0]=0; //can be omitted
		boundary1=_array_size/2;
		//boundaries[3]=_array_size; //can be omitted


		cudaSetDevice(0);
		ghost0=new ghost_vector<T>(std::vector<T>::begin(), std::vector<T>::begin()+boundary1);
		cudaSetDevice(1);
		ghost1=new ghost_vector<T>(std::vector<T>::begin()+boundary1, std::vector<T>::end());
	 
	 }

	explicit big_vector(Index_Type _array_size, T const &_val):
       	 std::vector<T>::vector(_array_size, _val),
       	 array_size(_array_size) {
	 
		//can move to list initialization
		//generate the following code
		//boundaries[0]=0; //can be omitted
		boundary1=_array_size/2;
		//boundaries[3]=_array_size; //can be omitted


		cudaSetDevice(0);
		ghost0=new ghost_vector<T>(std::vector<T>::begin(), std::vector<T>::begin()+boundary1);
		ghost0->show_cpu_memory();
		//ghost0->upload();
		cudaSetDevice(1);
		ghost1=new ghost_vector<T>(std::vector<T>::begin()+boundary1, std::vector<T>::end());
		ghost1->show_cpu_memory();
		//ghost1->upload();
	 
	 }

	~big_vector(){

		//make sure there is no memory leak on the cpu side

		//generate the following code
		cudaSetDevice(0);
		delete ghost0;
		cudaSetDevice(1);
		delete ghost1;
	}

	Index_Type array_size;

	//generate the following code
	ghost_vector<T>* ghost0;
	ghost_vector<T>* ghost1;
	Index_Type boundary1; //XPDL_NUM_OF_GPUS-1

	//coherence should use the same coherence interface, such as get_cpu_r_raw_ptr
	//for multi-gpu cases, the return value should be a zip_pointer or zip_iterator
	//you can check the implementation of boost zip iterator
	
	//you should think about the memory here, 
	//better not to have extra cost
	const zip_pointer<T>& get_cpu_r_raw_ptr(){
		zip_pointer<T> result;
		coherent_on_cpu_r(result);
		return result;
		//return this->std::vector<T>::data();
	}

	void coherent_on_cpu_r(zip_pointer<T> &_result){

		ghost0->coherent_on_cpu_r();
		_result.p0=ghost0->ptr();
		_result.size0=ghost0->array_size;

		_result.p=ghost0->ptr();
		_result.size=array_size;

		ghost1->coherent_on_cpu_r();
		_result.p1=ghost1->ptr();
		_result.size1=ghost1->array_size;
	}

	void coherent_on_gpu_r(zip_pointer<T> &_result){

		cudaSetDevice(0);
		ghost0->coherent_on_cpu_r();
		_result.p0=ghost0->gptr();
		_result.size0=ghost0->size();

		_result.p=ghost0->gptr();
		_result.size=array_size;

		cudaSetDevice(1);
		ghost1->coherent_on_cpu_r();
		_result.p1=ghost1->gptr();
	}
}; 
*/
/*}}}*/


//alghoush ghost_vector is equiped with device_id,
//but it will not abuse it,
//it only use it when absolutely necessary,
//such as constructor and desctructor
//for other cases, it will not use it,
//such as upload and get_gpu_r_raw_ptr,
//because it may stupidly call cudaSetDevice 
//for the same device_id for calling those functions,
//which is not necessary, and the user can do it smartly
template <class T, unsigned int _device_id, typename Index_Type=std::size_t>
struct ghost_vector {

	using iter_type=typename std::vector<T>::iterator;

	explicit ghost_vector(iter_type const _start_iter, iter_type const _end_iter):
		cpu_start_iter(_start_iter), cpu_end_iter(_end_iter),
		array_size(_end_iter-_start_iter),
		cpu_coherent_unit(true),
		gpu_coherent_unit(true),
		device_id(_device_id) {
			cudaSetDevice(device_id);
			gpu=new thrust::device_vector<T>(array_size); 
		}

	explicit ghost_vector(iter_type const _start_iter, iter_type const _end_iter, T const & _val):
		cpu_start_iter(_start_iter), cpu_end_iter(_end_iter),
		array_size(_end_iter-_start_iter),
		cpu_coherent_unit(true),
		gpu_coherent_unit(true),
		device_id(_device_id) {
			cudaSetDevice(device_id);
			gpu=new thrust::device_vector<T>(array_size, _val); 
		}

	~ghost_vector(){
		cudaSetDevice(device_id);
		delete gpu;
	}

	iter_type const cpu_start_iter;
	iter_type const cpu_end_iter;
	Index_Type array_size;
        bool cpu_coherent_unit;
        bool gpu_coherent_unit;
	unsigned int device_id;
	thrust::device_vector<T>* gpu;

	auto begin() -> decltype(cpu_start_iter) {return cpu_start_iter;}
	auto end() -> decltype(cpu_end_iter) {return cpu_end_iter;}

	T* ptr()  {return &(*cpu_start_iter);}
	T* gptr()  {return thrust::raw_pointer_cast(& (* gpu->begin() ) );}

        void upload() {
       	 thrust::copy(
       			 cpu_start_iter,
       			 cpu_end_iter,
       			 gpu->begin()
       		     );

        }
        void download() {
       	 thrust::copy(
       			 gpu->begin(),
       			 gpu->end(),
       			 cpu_start_iter
       		     );

        }

	void coherent_on_cpu_r(){
       	 if( !cpu_coherent_unit ){
       		 download();
       		 cpu_coherent_unit=true;
       	 }
	}

	void coherent_on_cpu_w(){
       	 cpu_coherent_unit=true;
       	 gpu_coherent_unit=false;
	}
	void coherent_on_cpu_rw(){
       	 if( !cpu_coherent_unit ){
       		 download();
       		 cpu_coherent_unit=true;
       	 }
       	 gpu_coherent_unit=false;
	}

	void coherent_on_gpu_r(){
       	 if( !gpu_coherent_unit ){
       		 upload();
       		 gpu_coherent_unit=true;
       	 }
	}
	void coherent_on_gpu_w(){
       	 gpu_coherent_unit=true;
       	 cpu_coherent_unit=false;
	}
	void coherent_on_gpu_rw(){
       	 if( !gpu_coherent_unit ){
       		 upload();
       		 gpu_coherent_unit=true;
       	 }
       	 cpu_coherent_unit=false;
	}


	void show_cpu_memory(){
		std::copy(cpu_start_iter, cpu_end_iter, std::ostream_iterator<T>(std::cout, " ") );
		std::cout<<std::endl;
	}

	//auto gbegin() 
	//auto gend() 

};

//algorithm that use big_vector
//must internally iterate over its several ghost_vector,
//intead of one global vector, even for cpu memory.
//thus no need to have coherent flags at big_vector level


template <class T, class Index_Type=std::size_t>
struct big_vector: public std::vector<T> {
	explicit big_vector(Index_Type _array_size):
       	 std::vector<T>::vector(_array_size),
       	 array_size(_array_size), VECTORPU_ALL_BOUNDARIES_LIST_ASSIGNMENT(XPDL_NUM_OF_GPUS), VECTORPU_ALL_GHOST_VECTOR_LIST_ASSIGNMENT(XPDL_NUM_OF_GPUS) {
	 
		//can move to list initialization
		//generate the following code
		//boundaries[0]=0; //can be omitted
		//boundary1=_array_size/2;
		//boundaries[3]=_array_size; //can be omitted
		
		//VECTORPU_ALL_BOUNDARIES_ASSIGNMENT(XPDL_NUM_OF_GPUS)


		//cudaSetDevice(0);
		//ghost0=new ghost_vector<T,0>(std::vector<T>::begin(), std::vector<T>::begin()+boundary0);
		//cudaSetDevice(1);
		//ghost1=new ghost_vector<T,1>(std::vector<T>::begin()+boundary0, std::vector<T>::end());
	 
	 }

	explicit big_vector(Index_Type _array_size, T const &_val):
       	 std::vector<T>::vector(_array_size, _val),
       	 array_size(_array_size), VECTORPU_ALL_BOUNDARIES_LIST_ASSIGNMENT(XPDL_NUM_OF_GPUS), VECTORPU_ALL_GHOST_VECTOR_LIST_ASSIGNMENT_WITH_VAL(XPDL_NUM_OF_GPUS) {

	 
		//can move to list initialization
		//generate the following code
		//boundaries[0]=0; //can be omitted
		//boundary1=_array_size/2;
		//boundaries[3]=_array_size; //can be omitted

		//VECTORPU_ALL_BOUNDARIES_ASSIGNMENT(XPDL_NUM_OF_GPUS)


		//cudaSetDevice(0);
		//ghost0=new ghost_vector<T,0>(std::vector<T>::begin(), std::vector<T>::begin()+boundary0, _val);
		//ghost0->show_cpu_memory();
		//ghost0->upload(); //if thrust can initialize by value, then we can save one operation and get higher performance
		//cudaSetDevice(1);
		//ghost1=new ghost_vector<T,1>(std::vector<T>::begin()+boundary0, std::vector<T>::end(), _val);
		//ghost1->show_cpu_memory();
		//ghost1->upload();
	 
	 }

	//when enabled, an error encountered, leave it for now
	//~big_vector(){

		//delete ghost0;
		//delete ghost1;
	//}

	Index_Type array_size;

	//Index_Type boundary0; //XPDL_NUM_OF_GPUS-1
	VECTORPU_ALL_BOUNDARIES_DECL(XPDL_NUM_OF_GPUS)

	//generate the following code
	//ghost_vector<T,0>* ghost0;
	//ghost_vector<T,1>* ghost1;
	VECTORPU_ALL_GHOST_VECTOR_DECL(XPDL_NUM_OF_GPUS)

	zip_pointer<T> result; //TODO: make this member exist when application use it

	//coherence should use the same coherence interface, such as get_cpu_r_raw_ptr
	//for multi-gpu cases, the return value should be a zip_pointer or zip_iterator
	//you can check the implementation of boost zip iterator
	
	//you should think about the memory here, 
	//better not to have extra cost
	const zip_pointer<T>& get_cpu_r_raw_ptr(){
		coherent_on_cpu_r();
		return result;
		//return this->std::vector<T>::data();
	}

	zip_pointer<T>& get_cpu_w_raw_ptr(){
		coherent_on_cpu_w();
		return result;
	}

	zip_pointer<T>& get_cpu_rw_raw_ptr(){
		coherent_on_cpu_rw();
		return result;
	}

	//Cont: more

	/*
	const zip_pointer<T>& get_gpu_r_raw_ptr(){
		//try to take the advantage of RVO,
		//but the behavior looks weird
		//this code heavily rely on optimization,
		//from O0 it is not correct, but O2 make it partially correct
		zip_pointer<T> result;
		coherent_on_gpu_r(result);
		return result;
	}  */


	const zip_pointer<T>& get_gpu_r_raw_ptr(){
		//Efficient but not working solution
		//----------------------------------//
		//zip_pointer<T> result;
		//coherent_on_gpu_r(result);
		//return result;
		//----------------------------------//

		//this code heavily rely on optimization,
		//from O0 it is not correct, but O2 make it partially correct
		//coherent_on_gpu_r(result);
		coherent_on_gpu_r();
		return result;
	}

	zip_pointer<T>& get_gpu_w_raw_ptr(){
		coherent_on_gpu_w();
		return result;
	}

	zip_pointer<T>& get_gpu_rw_raw_ptr(){
		coherent_on_gpu_rw();
		return result;
	}

		

	void coherent_on_cpu_r(){

		result.size=array_size;

		VECTORPU_ALL_GHOST_VECTOR_DECL_CPU(XPDL_NUM_OF_GPUS, r)

		//ghost0->coherent_on_cpu_r();
		//result.p0=ghost0->ptr();
		//result.size0=ghost0->array_size;


		//ghost1->coherent_on_cpu_r();
		//result.p1=ghost1->ptr();
		//result.size1=ghost1->array_size;
	}

	void coherent_on_cpu_w(){

		result.size=array_size;

		VECTORPU_ALL_GHOST_VECTOR_DECL_CPU(XPDL_NUM_OF_GPUS, w)

		//ghost0->coherent_on_cpu_w();
		//result.p0=ghost0->ptr();
		//result.size0=ghost0->array_size;


		//ghost1->coherent_on_cpu_w();
		//result.p1=ghost1->ptr();
		//result.size1=ghost1->array_size;
	}

	void coherent_on_cpu_rw(){

		result.size=array_size;

		VECTORPU_ALL_GHOST_VECTOR_DECL_CPU(XPDL_NUM_OF_GPUS, rw)

		//ghost0->coherent_on_cpu_rw();
		//result.p0=ghost0->ptr();
		//result.size0=ghost0->array_size;


		//ghost1->coherent_on_cpu_rw();
		//result.p1=ghost1->ptr();
		//result.size1=ghost1->array_size;
	}

	void coherent_on_gpu_r(){

		result.size=array_size;

		VECTORPU_ALL_GHOST_VECTOR_DECL_GPU(XPDL_NUM_OF_GPUS, r)

		//cudaSetDevice(0);
		//ghost0->coherent_on_gpu_r();
		//result.p0=ghost0->gptr();
		//result.size0=ghost0->array_size;

		//cudaSetDevice(1);
		//ghost1->coherent_on_gpu_r();
		//result.p1=ghost1->gptr();
		//result.size1=ghost1->array_size;
	}

	void coherent_on_gpu_w(){

		result.size=array_size;

		VECTORPU_ALL_GHOST_VECTOR_DECL_GPU(XPDL_NUM_OF_GPUS, w)

		//cudaSetDevice(0);
		//ghost0->coherent_on_gpu_w();
		//result.p0=ghost0->gptr();
		//result.size0=ghost0->array_size;

		//cudaSetDevice(1);
		//ghost1->coherent_on_gpu_w();
		//result.p1=ghost1->gptr();
		//result.size1=ghost1->array_size;
	}

	void coherent_on_gpu_rw(){

		result.size=array_size;

		VECTORPU_ALL_GHOST_VECTOR_DECL_GPU(XPDL_NUM_OF_GPUS, rw)

		//cudaSetDevice(0);
		//ghost0->coherent_on_gpu_rw();
		//result.p0=ghost0->gptr();
		//result.size0=ghost0->array_size;

		//cudaSetDevice(1);
		//ghost1->coherent_on_gpu_rw();
		//result.p1=ghost1->gptr();
		//result.size1=ghost1->array_size;
	}

};

#endif

#endif

/*}}}*/

#else

//cpu_vector
/*{{{*/

	template <class T, typename Index_Type=std::size_t>
	struct cpu_vector : public std::vector<T> {
		/*{{{*/
		explicit cpu_vector(Index_Type size): std::vector<T>::vector(size){}
   	        explicit cpu_vector(Index_Type _array_size, T const &val): std::vector<T>::vector(_array_size, val) {}

		const T& r(const Index_Type& i) {
			return *(std::vector<T>::begin()+i);
		}
		void w(const Index_Type& i, const T& new_val) {
			*(std::vector<T>::begin()+i)=new_val;
		}

		const T* get_r_raw(){ return this->std::vector<T>::data();  }
		T* get_w_raw(){ return this->std::vector<T>::data();  }
		Index_Type size(){ return this->std::vector<T>::size(); }

		typename std::vector<T>::const_iterator get_cpu_r_begin(){
			return std::vector<T>::begin();
		}
		typename std::vector<T>::const_iterator get_cpu_r_end(){
			return std::vector<T>::end();
		}
		/*}}}*/
	};
/*}}}*/

#endif


//Portability layer
/*{{{*/


#if !defined(XPDL_NUM_OF_GPUS) || XPDL_NUM_OF_GPUS==0
#warning("self_adaptive_vector=cpu_vector")
	template<class T>
	using self_adaptive_vector=cpu_vector<T>;
	template<class T>
	using vector=cpu_vector<T>;

#elif XPDL_NUM_OF_GPUS == 1
#warning("self_adaptive_vector=min_vector")

	template<class T>
	using vector=min_vector<T>;

	template<class T>
	using self_adaptive_vector=min_vector<T>;

#elif XPDL_NUM_OF_GPUS > 1
#warning("self_adaptive_vector=big_vector")
	template<class T>
	using self_adaptive_vector=big_vector<T>;

#endif

//template<class T>
//using vector=cpu_vector<T>;

//A great idea: if no GPU defined, then GW becomes W, likewise for other facility
//for pure CPU, it not only can run, but run without unnecessary overhead.
/*}}}*/


#if defined(XPDL_NUM_OF_GPUS) && (XPDL_NUM_OF_GPUS >=1)

//algorithm
/*{{{*/

//copy
/*{{{*/

#define VECTORPU_COPY_CG_META (RI)(REI)(GWI)
template<class T>
void copy(
		typename std::vector<T>::const_iterator _start, 
		typename std::vector<T>::const_iterator _end, 
		typename thrust::device_vector<T>::iterator _dest
	){

	thrust::copy(_start, _end, _dest);
}

#define VECTORPU_COPY_CC_META (RI)(REI)(WI)
template<class T>
void copy(
		typename std::vector<T>::const_iterator start,
		typename std::vector<T>::const_iterator end,
		typename std::vector<T>::iterator dest
	){
	std::copy(start, end, dest);
}
/*}}}*/

//fill
/*{{{*/

#define fill_c_flow (WI)(NA)(NA) 
template<class T, class Size_Type=std::size_t>
void fill(
		typename std::vector<T>::iterator _start, 
		typename std::vector<T>::iterator _end, 
		T val
		){
	std::fill(_start, _end, val);
}


#define fill_g_flow (GWI)(NA)(NA)
template<class T, class Size_Type=std::size_t>
void fill(
		typename thrust::device_vector<T>::iterator _start,
		typename thrust::device_vector<T>::iterator _end,
		T val
		){
	thrust::fill(_start, _end, val);
}


/*}}}*/

//for_each
/*{{{*/

#define VECTORPU_FOR_EACH_WI_META (WI)(NA)(NA)
#define VECTORPU_FOR_EACH_RWI_META (RWI)(NA)(NA)
template <class T, class Function, class Size_Type=std::size_t>
void for_each(
		typename std::vector<T>::iterator _start, 
		typename std::vector<T>::iterator _end, 
		//Size_Type size,
		Function f
		){
	std::for_each(_start, _end, f);
}

#define VECTORPU_FOR_EACH_RI_META (RI)(NA)(NA)
template <class T, class Function, class Size_Type=std::size_t>
void for_each(
		typename std::vector<T>::const_iterator _start, 
		typename std::vector<T>::const_iterator _end, 
		//Size_Type size,
		Function f
		){
	std::for_each(_start, _end, f);
}

#define VECTORPU_FOR_EACH_GWI_META (GWI)(NA)(NA)
#define VECTORPU_FOR_EACH_GRWI_META (GRWI)(NA)(NA)
template <class T, class Function, class Size_Type=std::size_t>
void for_each(
		typename thrust::device_vector<T>::iterator _start,
		typename thrust::device_vector<T>::iterator _end,
		//Size_Type size,
		Function f
		){
	thrust::for_each(_start, _end, f);
}

#define VECTORPU_FOR_EACH_GRI_META (GRI)(NA)(NA)
template <class T, class Function, class Size_Type=std::size_t>
void for_each(
		typename thrust::device_vector<T>::const_iterator _start,
		typename thrust::device_vector<T>::const_iterator _end,
		//Size_Type size,
		Function f
		){
	thrust::for_each(_start, _end, f);
}

#ifdef XPDL_NUM_OF_GPUS
#if XPDL_NUM_OF_GPUS > 1

//template <class T, class Function, class Size_Type=std::size_t>
//void for_each(
		//zip_iterator<T> start,
		//Size_Type size,
		//Function f
		//){
//}

#endif
#endif

/*}}}*/


/*}}}*/

#endif

//vectorpu macro library
/*{{{*/

//if not define XPDL_NUM_OF_GPUS, then big_vector is not enabled,
//then we need the header
//if less than 1 GPU by XPDL_NUM_OF_GPUS, then big_vector's macro library is not enabled either,
//then we need the header
#if !defined(XPDL_NUM_OF_GPUS) || ( defined(XPDL_NUM_OF_GPUS) && XPDL_NUM_OF_GPUS <= 1)

#undef BOOST_PP_VARIADICS
#define BOOST_PP_VARIADICS 1
#include <boost/preprocessor.hpp>

#endif


/* Common facility */
/*{{{*/

//NA is much less probable to collide with username, previously N is too easy to collide
#define NA_VALUE 0  

#define R_VALUE 1
#define W_VALUE 2
#define RW_VALUE 3
#define GR_VALUE 4
#define GW_VALUE 5
#define GRW_VALUE 6

#define RI_VALUE 7
#define WI_VALUE 8
#define RWI_VALUE 9
#define GRI_VALUE 10
#define GWI_VALUE 11
#define GRWI_VALUE 12


#define VECTORPU_META_DATA(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)
#define VECTORPU_ALL_SIZE BOOST_PP_SEQ_SIZE(VECTORPU_DESCRIBE)

/*}}}*/

/* For one-time use function */
/*{{{*/

#if defined(XPDL_NUM_OF_GPUS) && XPDL_NUM_OF_GPUS==1

#define R(x)   x.get_cpu_r_raw_ptr()
#define W(x)   x.get_cpu_w_raw_ptr()
#define RW(x)  x.get_cpu_rw_raw_ptr()
#define GR(x)  x.get_gpu_r_raw_ptr()
#define GW(x)  x.get_gpu_w_raw_ptr()
#define GRW(x) x.get_gpu_rw_raw_ptr()

#define SR(x)   x.coherent_on_cpu_r()
#define SW(x)   x.coherent_on_cpu_w()
#define SRW(x)  x.coherent_on_cpu_rw()
#define SGR(x)  x.coherent_on_gpu_r()
#define SGW(x)  x.coherent_on_gpu_w()
#define SGRW(x) x.coherent_on_gpu_rw()

#define RI(x)   x.get_cpu_r_begin()
#define WI(x)   x.get_cpu_w_begin()
#define RWI(x)  x.get_cpu_rw_begin()
#define GRI(x)  x.get_gpu_r_begin()
#define GWI(x)  x.get_gpu_w_begin()
#define GRWI(x) x.get_gpu_rw_begin()

#define REI(x)   x.get_cpu_r_end()
#define WEI(x)   x.get_cpu_w_end()
#define RWEI(x)  x.get_cpu_rw_end()
#define GREI(x)  x.get_gpu_r_end()
#define GWEI(x)  x.get_gpu_w_end()
#define GRWEI(x) x.get_gpu_rw_end()

//only transfer back the first few elements, don't care about coherence anymore
//good for the last step of reduce, where you only want to transfer one value back
//Partial Read
#define PR(x, _size) x.get_cpu_r_raw_ptr_no_management(_size)

#else

#define R(x)   x.get_r_raw()
#define W(x)   x.get_w_raw()
#define RI(x)   x.get_cpu_r_begin()
#define REI(x)   x.get_cpu_r_end()
//if no GPU, no GR should appear
//#define GR(x)  x.get_r_raw()
//#define GW(x)  x.get_w_raw()

#endif

/*}}}*/

/* For normal visible function */
/*{{{*/

#define NA_WRAP(x) x

#define R_WRAP(x) R(x)
#define W_WRAP(x) W(x)
#define RW_WRAP(x) RW(x)
#define GR_WRAP(x) GR(x)
#define GW_WRAP(x) GW(x)
#define GRW_WRAP(x) GRW(x)

#define RI_WRAP(x)   RI(x)  
#define WI_WRAP(x)   WI(x)  
#define RWI_WRAP(x)  RWI(x) 
#define GRI_WRAP(x)  GRI(x) 
#define GWI_WRAP(x)  GWI(x) 
#define GRWI_WRAP(x) GRWI(x)



#define STRIP_PARENTHESIS(...) __VA_ARGS__
#define RUN(...) __VA_ARGS__

#define QUERY_STRIP( new_format, n )   STRIP_PARENTHESIS BOOST_PP_SEQ_ELEM(n, new_format)
#define GET_FUNCTION_NAME( new_format )   BOOST_PP_SEQ_ELEM(0, new_format)

#define WRAP_ONE_PARAMETER(meta_data, runtime_arg) BOOST_PP_CAT(meta_data, _WRAP) (runtime_arg)

#define HANDLE_ONE_ELEMNT_IN_ARRAY(n, text) WRAP_ONE_PARAMETER( BOOST_PP_SEQ_ELEM(n,BOOST_PP_TUPLE_ELEM(3, 0, text)), BOOST_PP_SEQ_ELEM(n,BOOST_PP_TUPLE_ELEM(3, 1, text)) ) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL (n, BOOST_PP_DEC( BOOST_PP_TUPLE_ELEM(3, 2, text) ) ) )
#define ADAPTER_HANDLE_ONE_ELEMNT_IN_ARRAY(z,n,text) HANDLE_ONE_ELEMNT_IN_ARRAY(n, text)
#define GET_ANNOTATED_PARAMETERS( new_format )   BOOST_PP_REPEAT( BOOST_PP_SEQ_SIZE( BOOST_PP_CAT( BOOST_PP_SEQ_ELEM(0, new_format), _flow  )     ) , ADAPTER_HANDLE_ONE_ELEMNT_IN_ARRAY, (BOOST_PP_CAT( BOOST_PP_SEQ_ELEM(0, new_format), _flow) , BOOST_PP_VARIADIC_TO_SEQ( RUN(QUERY_STRIP( new_format, 2 )) ), BOOST_PP_SEQ_SIZE( BOOST_PP_CAT( BOOST_PP_SEQ_ELEM(0, new_format), _flow  )  ) ) )
#define GET_CUSTOM_ANNOTATED_PARAMETERS( new_format )   BOOST_PP_REPEAT( BOOST_PP_SEQ_SIZE( BOOST_PP_SEQ_ELEM(3, new_format)     ) , ADAPTER_HANDLE_ONE_ELEMNT_IN_ARRAY, ( BOOST_PP_SEQ_ELEM(3, new_format) , BOOST_PP_VARIADIC_TO_SEQ( RUN(QUERY_STRIP( new_format, 2 )) ), BOOST_PP_SEQ_SIZE( BOOST_PP_SEQ_ELEM(3, new_format)  ) ) )

#define CALL( new_format )  GET_FUNCTION_NAME(new_format) RUN( QUERY_STRIP( new_format, 1)  ) ( GET_ANNOTATED_PARAMETERS(new_format) )
#define CALLC( new_format ) GET_FUNCTION_NAME(new_format) RUN( QUERY_STRIP( new_format, 1)) ( GET_CUSTOM_ANNOTATED_PARAMETERS(new_format) )




/*}}}*/

/* For lambda function */
/*{{{*/


#define VECTORPU_LAMBDA_LOOP_SIZE BOOST_PP_DIV(VECTORPU_ALL_SIZE,3)

#define CONST_OR_NOT(access) BOOST_PP_IF( BOOST_PP_EQUAL( BOOST_PP_CAT(access, _VALUE), R_VALUE), const, )

#define VECTORPU_TEMP_DEFINE(type, var, access) CONST_OR_NOT(access) type* BOOST_PP_CAT(var,_temp) = access(var);

#define VECTORPU_TEMP_DEFINE_HANDLE_THREE_ELEM(n) \
	VECTORPU_TEMP_DEFINE(  \
			BOOST_PP_SEQ_ELEM(  BOOST_PP_MUL(n,3)  ,VECTORPU_DESCRIBE), \
		       	BOOST_PP_SEQ_ELEM(  BOOST_PP_ADD(BOOST_PP_MUL(n,3),1)  ,VECTORPU_DESCRIBE), \
			BOOST_PP_SEQ_ELEM(  BOOST_PP_ADD(BOOST_PP_MUL(n,3),2)  ,VECTORPU_DESCRIBE)  \
			)

#define VECTORPU_ADAPTER_VECTORPU_TEMP_DEFINE_HANDLE_THREE_ELEM(z, n, text) VECTORPU_TEMP_DEFINE_HANDLE_THREE_ELEM(n)

#define ALL_TEMP_DEFINE BOOST_PP_REPEAT(VECTORPU_LAMBDA_LOOP_SIZE, VECTORPU_ADAPTER_VECTORPU_TEMP_DEFINE_HANDLE_THREE_ELEM, )


#define VECTORPU_LAMBDA_ONE_ARGUMENT(type, var, access) BOOST_PP_CAT(var,_temp)

#define VECTORPU_LAMBDA_ONE_ARGUMENT_HANDLE_THREE_ELEM(n) \
	VECTORPU_LAMBDA_ONE_ARGUMENT(  \
			BOOST_PP_SEQ_ELEM(  BOOST_PP_MUL(n,3)  ,VECTORPU_DESCRIBE), \
		       	BOOST_PP_SEQ_ELEM(  BOOST_PP_ADD(BOOST_PP_MUL(n,3),1)  ,VECTORPU_DESCRIBE), \
			BOOST_PP_SEQ_ELEM(  BOOST_PP_ADD(BOOST_PP_MUL(n,3),2)  ,VECTORPU_DESCRIBE)  \
			) \
 	BOOST_PP_COMMA_IF( BOOST_PP_NOT_EQUAL(n, BOOST_PP_DEC( VECTORPU_LAMBDA_LOOP_SIZE ) ))

#define VECTORPU_ADAPTER_VECTORPU_LAMBDA_ONE_ARGUMENT_HANDLE_THREE_ELEM(z, n, text) VECTORPU_LAMBDA_ONE_ARGUMENT_HANDLE_THREE_ELEM(n)

#define VECTORPU_LAMBDA_ALL_ARGUMENTS BOOST_PP_REPEAT(VECTORPU_LAMBDA_LOOP_SIZE, VECTORPU_ADAPTER_VECTORPU_LAMBDA_ONE_ARGUMENT_HANDLE_THREE_ELEM, )


#define EXPAND_Y_AND_LAMBDA_REPLACE(y) lambda(y) 

#define VECTORPU_LAMBDA_GEN \
	{ \
		ALL_TEMP_DEFINE \
		EXPAND_Y_AND_LAMBDA_REPLACE(VECTORPU_LAMBDA_ALL_ARGUMENTS) \
	}
/*}}}*/

/*}}}*/

}



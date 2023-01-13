BMP_FILE=biomutant.bmp

build: build_openmp build_mpi build_pthreads build_hybrid1 build_hybrid2

run: run_openmp run_mpi run_pthreads run_hybrid1 run_hybrid2

build_secvential:
	g++ -o secvential-sobel secvential-sobel.cpp -Wall

run_secvential: build_secvential
	./secvential-sobel ${BMP_FILE}

build_openmp:
	g++ -fopenmp -o openmp-sobel openmp-sobel.cpp

run_openmp: build_openmp
	./openmp-sobel ${BMP_FILE}

build_mpi:
	mpic++ -o mpi-sobel mpi-sobel.cpp

run_mpi: build_mpi
	mpirun -np 6 mpi-sobel ${BMP_FILE}

build_pthreads:
	g++ -o pthreads-sobel pthreads-sobel.cpp -lpthread -Wall

run_pthreads: build_pthreads
	./pthreads-sobel ${BMP_FILE}

build_hybrid1:
	mpic++ -fopenmp -o hybrid1-sobel hybrid1-sobel.cpp

run_hybrid1: build_hybrid1
	mpirun -np 6 hybrid1-sobel ${BMP_FILE}

build_hybrid2:
	mpic++ -o hybrid2-sobel hybrid2-sobel.cpp -lpthread

run_hybrid2: build_hybrid2
	mpirun -np 4 hybrid2-sobel ${BMP_FILE}

clean:
	rm -f secvential-sobel openmp-sobel mpi-sobel pthreads-sobel hybrid1-sobel hybrid2-sobel sobel_*.bmp

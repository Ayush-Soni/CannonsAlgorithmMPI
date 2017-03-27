#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#define MPICW MPI_COMM_WORLD
using namespace std;

int main(int argc, char* argv[]) {

	//rank, size and dimension, common to all processes
	int rank, size, dimension, i, j;

	//input and output matrices
	double *buffer, *matrixA, *matrixB, *matrixC, resPij = 0.0, currentElementFromA, currentElementFromB, tempA, tempB;

	//MPI interfaces to initalize environment and get rank, size. [DEBUG]
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPICW, &rank);
	MPI_Comm_size(MPICW, &size);
	MPI_Status status;
	MPI_Request request, reqSendA, reqSendB, reqRecv;

	//Initializing dimension as sqrt(size) because the number of processes = total number of elements in either of the input matrix
	dimension = (int)sqrt(size);
	int sizeOfBuffer = 2 * dimension*dimension;
	matrixA = (double*)calloc(dimension*dimension, sizeof(double));
	matrixB = (double*)calloc(dimension*dimension, sizeof(double));
	matrixC = (double*)calloc(dimension*dimension, sizeof(double));
	buffer = (double*)calloc(sizeOfBuffer, sizeof(double));
	
	//Initializing buffer
	MPI_Buffer_attach((void*)buffer, sizeOfBuffer* sizeof(double));

	//Taking input of the master matrices
	if (rank == 0) {
		printf("\nMatrix A[%d][%d]:\n", dimension, dimension);
		//printf("\nEnter Matrix A[%d][%d]:", dimension, dimension);
		for (i = 0; i<dimension; i++) {
			for (j = 0; j<dimension; j++) {
				matrixA[i*dimension + j] = i*dimension + j + 1;
				//cin >> matrixA[i*dimension + j];
				cout<< matrixA[i*dimension + j]<<" ";
			}
			cout <<"\n";
		}
		fflush(stdout);
		printf("\nEnter Matrix B[%d][%d]:\n", dimension, dimension);
		for (i = 0; i<dimension; i++) {
			for (j = 0; j<dimension; j++) {
				matrixB[i*dimension + j] = dimension*dimension - i*dimension - j;
				//cin >> matrixB[i*dimension + j];
				cout<< matrixB[i*dimension + j]<<" ";
			}
			cout << "\n";
		}
		fflush(stdout);
	}


	//Input matrices are now available, moving to initial setup stage
	//Giving each processing element P(i,j), the elements A(i,j) and B(i,j) [DEBUG]
	MPI_Scatter(matrixA, 1, MPI_DOUBLE, &currentElementFromA, 1, MPI_DOUBLE, 0, MPICW);
	MPI_Scatter(matrixB, 1, MPI_DOUBLE, &currentElementFromB, 1, MPI_DOUBLE, 0, MPICW);


	//Initial setup is now complete
	//Proceeding towards intermediate setup
	//Performing shifts: Row-wise for A, Col-wise for B
	

	//Shifts for A: [DEBUG]
	int myRow = floor((double)(rank / dimension));
	int destinationRankRow = floor((double)((rank - myRow) / dimension))<myRow ? (rank - myRow + dimension) : (rank - myRow);
	int sourceRankRow = floor((double)((rank + myRow) / dimension))>myRow ? (rank + myRow - dimension) : (rank + myRow);
	MPI_Ibsend(&currentElementFromA, 1, MPI_DOUBLE, destinationRankRow, rank, MPICW, &request);
	MPI_Recv(&tempA, 1, MPI_DOUBLE, sourceRankRow, sourceRankRow, MPICW, MPI_STATUSES_IGNORE);
	currentElementFromA = tempA;

	//Shifts for B: [DEBUG]
	int myCol = rank%dimension;
	int destinationRankCol = (rank - myCol*dimension)<0 ? (rank - myCol*dimension + dimension*dimension) : (rank - myCol*dimension);
	int sourceRankCol = (rank + myCol*dimension)>=dimension*dimension ? (rank + myCol*dimension - dimension*dimension) : (rank + myCol*dimension);
	MPI_Ibsend(&currentElementFromB, 1, MPI_DOUBLE, destinationRankCol, rank, MPICW, &request);
	MPI_Recv(&tempB, 1, MPI_DOUBLE, sourceRankCol, sourceRankCol, MPICW, MPI_STATUSES_IGNORE);
	currentElementFromB = tempB;

	//Testing A and B:
	printf("\nRank:%d, currentElementFromA:%lf, currentElementFromB:%lf", rank, currentElementFromA, currentElementFromB);
	
	//Assemble at this point
	fflush(stdout);
	MPI_Barrier(MPICW);


	//Algorithm [DEBUG]
	int sourceRankA = floor((double)((rank + 1) / dimension))>floor((double)(rank / dimension)) ? (rank + 1 - dimension) : (rank + 1);
	int destinationRankA = ((floor((double)((rank - 1) / dimension))<floor((double)(rank / dimension)))||(rank-1<0))? (rank - 1 + dimension) : (rank - 1);
	int sourceRankB = (rank + dimension)>=dimension*dimension ? (rank + dimension - dimension*dimension) : (rank + dimension);
	int destinationRankB = (rank - dimension)<0 ? (rank - dimension + dimension*dimension) : (rank - dimension);

	MPI_Barrier(MPICW);
	//Circular shifts, dimension times
	for(i=0;i<dimension;i++) {
		
		//
		resPij+=currentElementFromA*currentElementFromB;
		if (rank == 0) {
			cout << "\nIteration:" << i + 1;
		}
		//ring-rotate A 1 time leftwards
		MPI_Ibsend(&currentElementFromA, 1, MPI_DOUBLE, destinationRankA, rank, MPICW, &reqSendA);
		MPI_Recv(&tempA, 1, MPI_DOUBLE, sourceRankA, sourceRankA, MPICW, &status);
		cout << "\n->Rank" << rank << ", tempA:" << tempA;
		fflush(stdout);
		MPI_Barrier(MPICW);
		//ring-rotate B 1 time upwards
		MPI_Ibsend(&currentElementFromB, 1, MPI_DOUBLE, destinationRankB, rank+dimension*dimension, MPICW, &reqSendB);
		MPI_Recv(&tempB, 1, MPI_DOUBLE, sourceRankB, sourceRankB+dimension*dimension, MPICW, &status);
		cout << "\n->Rank" << rank << ", tempB:" << tempB;
		fflush(stdout);
		MPI_Barrier(MPICW);
	
		currentElementFromA = tempA;
		currentElementFromB = tempB;
		cout << "\n-->Rank:" << rank << ", resPij:" << resPij;
		fflush(stdout);
		MPI_Barrier(MPICW);
		
	}

	MPI_Barrier(MPICW);
	int bufSize = sizeOfBuffer * sizeof(double);
	MPI_Buffer_detach(buffer, &bufSize);

	//Final Answer:
	//Needs to be gathered	[DEBUG]
	MPI_Gather(&resPij, 1, MPI_DOUBLE, matrixC, 1, MPI_DOUBLE, 0, MPICW);

	fflush(stdout);
	MPI_Barrier(MPICW);
	if(rank==0) {
		cout << "\nResult:";
		for (i = 0; i < dimension; i++) {
			cout << "\n";
			for (j = 0; j < dimension; j++) {
				cout << matrixC[i*dimension + j]<< " ";
			}
		}
	}


	//The End
	MPI_Finalize();
	return 0;
}
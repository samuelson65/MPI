#include <mpi.h>

// "number blocks" in the struct
static const int NUM_STRUCT_FIELDS = 3;
void defineStruct_pass_in_data(MPI_Datatype* typeToBeCreated);
void define_op_max_pass_in_data(MPI_Op* op);
void max_location_index(void *in, void *inout, int *len, MPI_Datatype *type);
/*
  structure for MPI to pass in one column of data
*/
typedef struct pass_in_data_struc{
  double val;
  int rank;
  int index;
} pass_in_data;

void max_location_index(void *in, void *inout, int *len, MPI_Datatype *type){
    pass_in_data *invals    = (pass_in_data*)in;
    pass_in_data *inoutvals = (pass_in_data*)inout;

    for (int i=0; i<*len; i++) {
        if (invals[i].val > inoutvals[i].val) {
            inoutvals[i].val  = invals[i].val;
            inoutvals[i].rank = invals[i].rank;
            inoutvals[i].index = invals[i].index;
        }
    }
    return;
}


void defineStruct_pass_in_data(MPI_Datatype* typeToBeCreated){
  /* create our new data type and its associated methods */

  MPI_Datatype types[NUM_STRUCT_FIELDS];
  MPI_Aint displ[NUM_STRUCT_FIELDS];
  int blklen[NUM_STRUCT_FIELDS];

	types[0] = MPI_DOUBLE; // type of val
	types[1] = MPI_INT;   // type of rank
	types[2] = MPI_INT; // base type of index

  //if it's an array, then it has length >= 1
	blklen[0] = 1; // length of val
	blklen[1] = 1; // length of rank
	blklen[2] = 1; // length of index


  displ[0] = 0;
  pass_in_data sample;
  MPI_Aint base; // base address of a pass_in_data instance
  MPI_Get_address(&sample.val, &base);
  // base address of successive fields of a given pass_in_data instance:
  MPI_Aint oneField;
  MPI_Get_address(&sample.rank, &oneField);
  displ[1] = oneField - base; // offset (displacement) to 'y'
  MPI_Get_address(&sample.index, &oneField);
  displ[2] = oneField - base; // offset (displacement) to 'z'


  MPI_Type_create_struct(NUM_STRUCT_FIELDS, blklen, displ, types, typeToBeCreated);
  MPI_Type_commit(typeToBeCreated);
}

void define_op_max_pass_in_data(MPI_Op* op){
  MPI_Op_create(max_location_index, 1, op);
}



/*
void cleanUp_mpi(MPI_Datatype* typeToBeCreated){
  MPI_Op_free(&mpi_max_location_index);
  MPI_Type_free(&mpi_val_rank_pos);
}*/


// Below comment is from http://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.1/node80.htm

/*
function is the user-defined function, which must have the following four arguments: invec, inoutvec, len and datatype.
The ANSI-C prototype for the function is the following.
typedef void MPI_User_function( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);

The datatype argument is a handle to the data type that was passed into the call to MPI_REDUCE.
The user reduce function should be written such that the following holds: Let u[0], ... , u[len-1] be the len elements
in the communication buffer described by the arguments invec, len and datatype when the function is invoked;
let v[0], ... , v[len-1] be len elements in the communication buffer described by the arguments inoutvec,
len and datatype when the function is invoked; let w[0], ... , w[len-1] be len elements in the communication buffer
described by the arguments inoutvec, len and datatype when the function returns; then w[i] = u[i]°v[i],
for i=0 , ... , len-1, where ° is the reduce operation that the function computes.

Informally, we can think of invec and inoutvec as arrays of len elements that function is combining.
The result of the reduction over-writes values in inoutvec, hence the name.
Each invocation of the function results in the pointwise evaluation of the reduce operator on len elements:
I.e, the function returns in inoutvec[i] the value invec[i] ° inoutvec[i], for i=0, ... , count-1, where ° is the combining operation computed by the function.
*/


/*
MPI_OP_CREATE binds a user-defined global operation to an op handle that can subsequently be used in MPI_REDUCE, MPI_ALLREDUCE, MPI_REDUCE_SCATTER, and
MPI_SCAN. The user-defined operation is assumed to be associative. If commute = true, then the operation should be both commutative and associative.
If commute = false, then the order of operands is fixed and is defined to be in ascending, process rank order, beginning with process zero.
The order of evaluation can be changed, talking advantage of the associativity of the operation.
If commute = true then the order of evaluation can be changed, taking advantage of commutativity and associativity.
*/
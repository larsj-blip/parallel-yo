#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)
#define IS_MOTHER_PROCESS (rank == 0)

typedef int64_t int_t;
typedef double real_t;
typedef struct
{
    int x, y, subgrid_row_size, subgrid_column_size;
    real_t value;
} subgrid;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t
    *temp[2] = {NULL, NULL},
    *thermal_diffusivity,
    dt;

#define T(x, y) temp[0][(y) * (local_M + 2) + (x)]
#define T_next(x, y) temp[1][((y) * (local_M + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x, y) thermal_diffusivity[(y) * (local_M + 2) + (x)]

void time_step(void);
void boundary_condition(void);
void border_exchange(void);
void domain_init(void);
void domain_save(int_t iteration);
void domain_finalize(void);

int rank,
    amount_of_processes,
    dimensions[2] = {0, 0}, // letting mpi set this
    communicator_should_not_periodisize[2] = {0, 0},
    local_N,
    local_M,
    location_in_grid[2];

MPI_Comm cartesian_communicator;

MPI_Datatype custom_data_type;
MPI_Datatype writable_area;

void swap(real_t **m1, real_t **m2)
{
    real_t *tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}

int main(int argc, char **argv)
{
    // TODO 1:
    // - Initialize and finalize MPI.
    // - Create a cartesian communicator.
    // - Parse arguments in the rank 0 processes
    //   and broadcast to other processes

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &amount_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (IS_MOTHER_PROCESS)
    {

        OPTIONS *options = parse_args(argc, argv);
        if (!options)
        {
            fprintf(stderr, "Argument parsing failed\n");
            exit(1);
        }

        M = options->M;
        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_frequency, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Dims_create(amount_of_processes, 2, dimensions);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, communicator_should_not_periodisize, 1, &cartesian_communicator);
    domain_init();
    printf("dimensions: %d, %d", dimensions[0], dimensions[1]);

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        // TODO 6: Implement border exchange.
        // Hint: Creating MPI datatypes for rows and columns might be useful.
        border_exchange();

        boundary_condition();

        time_step();

        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);
            domain_save(iteration);
        }

        swap(&temp[0], &temp[1]);
    }

    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();

    MPI_Finalize();

    exit(EXIT_SUCCESS);
}

void time_step(void)
{
    real_t c, t, b, l, r, K, new_value;

    // TODO 3: Update the area of iteration so that each
    // process only iterates over its own subgrid.

    for (int_t y = 1; y <= local_M; y++)
    {
        for (int_t x = 1; x <= local_N; x++)
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}

void border_exchange(void)
{
    MPI_Datatype column, row;
//real_t before = T(1, local_N +1);
#define indexing_address_T(x, y) (y) * (local_M + 2) + (x)
    int left_node, right_node, up_node, down_node;
    MPI_Cart_shift(cartesian_communicator,
                   0,
                   1,
                   &up_node,
                   &down_node);
    MPI_Cart_shift(cartesian_communicator,
                   1,
                   1,
                   &left_node,
                   &right_node);

    MPI_Type_contiguous(local_N + 2, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Type_vector(local_M + 2, 1, local_N + 2, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    MPI_Sendrecv(temp[0] + indexing_address_T(0, 1),
                 1,
                 row,
                 up_node,
                 0,
                 temp[0] + indexing_address_T(0, local_N + 1),
                 1,
                 row,
                 down_node,
                 0,
                 cartesian_communicator,
                 MPI_STATUS_IGNORE);

    MPI_Sendrecv(temp[0] + indexing_address_T(0, local_N),
                 1,
                 row,
                 down_node,
                 1,
                 temp[0],
                 1,
                 row,
                 up_node,
                 1,
                 cartesian_communicator,
                 MPI_STATUS_IGNORE);

    MPI_Sendrecv(temp[0] + indexing_address_T(1, 0),
                 1,
                 column,
                 left_node,
                 2,
                 temp[0] + indexing_address_T(local_N+1, 0),
                 1,
                 column,
                 right_node,
                 2,
                 cartesian_communicator, MPI_STATUS_IGNORE);
    MPI_Sendrecv(temp[0] + indexing_address_T(local_N, 0),
                 1,
                 column,
                 right_node,
                 3,
                 temp[0],
                 1,
                 column,
                 left_node,
                 3,
                 cartesian_communicator, MPI_STATUS_IGNORE);

//    real_t after = T(1, local_N +1);
//    printf("before %f after %f rank %d\n", before, after, rank);
}

void boundary_condition(void)
{
    // TODO 4: Change the application of boundary conditions
    // to match the cartesian topology. communicate with above/below, right/left,
    // use built in handling of sending messages to places outside bounds of plane.
    // get subgrids that are at the borders, do the thing
    int location_in_grid[2];

    int left_node, right_node, up_node, down_node;

    MPI_Cart_shift(cartesian_communicator,
                   0,
                   1,
                   &up_node,
                   &down_node);

    MPI_Cart_shift(cartesian_communicator,
                   1,
                   1,
                   &left_node,
                   &right_node);

    MPI_Cart_coords(cartesian_communicator, rank, 2, location_in_grid);

    if (left_node == MPI_PROC_NULL)
    {
        for (int_t y = 1; y <= local_M; y++)
        {
            T(0, y) = T(2, y);
        }
    }
    if (right_node == MPI_PROC_NULL)
    {
        for (int_t y = 1; y <= local_M; y++)
        {
            T(local_N + 1, y) = T(local_N - 1, y);
        }
    }
    if (down_node == MPI_PROC_NULL)
    {
        for (int_t x = 1; x <= local_N; x++)
        {
            T(x, local_M + 1) = T(x, local_M - 1);
        }
    }
    if (up_node == MPI_PROC_NULL)
    {
        for (int_t x = 1; x <= local_N; x++)
        {
            T(x, 0) = T(x, 2);
        }
    }
    // printf(" rank: %d, l r u d  %d %d %d %d \n", rank, left_node, right_node, up_node, down_node );
}

void domain_init(void)
{
    // TODO 2:
    // - Find the number of columns and rows in each process' subgrid.
    // - Allocate memory for each process' subgrid.
    // - Find each process' offset to calculate the correct initial values.
    // Hint: you can get useful information from the cartesian communicator.
    // Note: you are allowed to assume that the grid size is divisible by
    // the number of processes.
    int
        offset_M,
        offset_N;
    MPI_Cart_coords(cartesian_communicator, rank, 2, location_in_grid);
    local_M = M / dimensions[0];
    local_N = N / dimensions[1];
    offset_M = local_M * location_in_grid[0];
    offset_N = local_N * location_in_grid[1];
    printf("offm, %d, offn, %d \n", offset_M, offset_N);
    printf("locm, locn: %d, %d \n", location_in_grid[0], location_in_grid[1]);

    temp[0] = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));
    temp[1] = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));
    thermal_diffusivity = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));

    dt = 0.1;

    for (int_t y = 1; y <= local_M; y++)
    {
        for (int_t x = 1; x <= local_N; x++)
        {
            real_t temperature = 30 + 30 * sin(((x + offset_N) + (y + offset_M)) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - (x + offset_N) + (y + offset_M)) / 20.0)) / 605.0;

            T(x, y) = temperature;
            T_next(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }
}

void domain_save(int_t iteration)
{
    // TODO 5: Use MPI I/O to save the state of the domain to file.
    // Hint: Creating MPI datatypes might be useful.
    int location_in_grid[2],
        offset_M,
        offset_N;

    MPI_Cart_coords(cartesian_communicator, rank, 2, location_in_grid);
    offset_M = local_M * location_in_grid[0];
    offset_N = local_N * location_in_grid[1];

    int global_size[2] = {M, N};
    int local_size[2] = {local_M, local_N};
    int local_origin[2] = {offset_M, offset_N};
    int local_size_with_halo[2] = {local_M + 2, local_N + 2};
    int origin_for_values_within_halo[2] = {1, 1};

    MPI_Datatype values_without_halo;
    MPI_Type_create_subarray(
        2, local_size_with_halo, local_size, origin_for_values_within_halo,
        MPI_ORDER_C, MPI_DOUBLE, &values_without_halo);
    MPI_Type_commit(&values_without_halo);

    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);
    int mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    MPI_File file;


    MPI_Type_create_subarray(2, global_size, local_size, local_origin,
        MPI_ORDER_C, MPI_DOUBLE, &writable_area);
    MPI_Type_commit(&writable_area);

    if (MPI_File_open(MPI_COMM_WORLD, filename, mode, MPI_INFO_NULL, &file) != MPI_SUCCESS)
    {
        printf("[MPI process %d] Failure in opening the file.\n", rank);

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_File_set_view(
        file, 0, MPI_DOUBLE, writable_area, "native", MPI_INFO_NULL);

    MPI_File_write_all(
        file, temp[0], 1, values_without_halo, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
}

void domain_finalize(void)
{
    free(temp[0]);
    free(temp[1]);
    free(thermal_diffusivity);
}

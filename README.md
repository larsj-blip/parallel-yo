# TDT4200 Problem set 2: More on MPI

## 2D heat simulation
In this exercise you will take a seqential implementation of the Finite Difference Method (FDM) for solving the 2D heat equation and write a distributed version using the Message Passing Interface (MPI). The sequential implementation is provided. Information on solving this assignment is described in the lecture slides and in the problem set description.

A sequential implementation is provided can be found in `heat_sequential.c` and should be kept as a reference. A copy of the sequential solution can be found in `heat_parallel.c`, in which you should write your parallel implementation.

## Run
### Setup
`make setup`

Creates folders `data`, `plots` and `video`.
- `data`: contains output from the simulation
- `plots`: contains output from plotting
- `video`: contains output from video generation

Prepares the check.

### Sequential solution
**Compile**

`make sequential`

**Run**

`./sequential -m [y_size] -n [x_size] -i [max_iteration] -s [snapshot_frequency]`

**Example**

```
make sequential
./sequential -m 256 -n 256 -i 100000 -s 1000
```

**Compile and run**

You can also execute both of the above commands together with default values with `make run_sequential`

### Parallel solution
**Compile**

`make parallel`

**Run**

`mpirun -np [number of MPI processes] [--oversubscribe] ./parallel -m [y_size] -n [x_size] -i [max_iteration] -s [snapshot_frequency]`

**!** MPI will complain that there are "not enough slots available" if you try to run with more processes than there are available processors. Passing the `--oversubscribe` option to `mpirun` will circumvent this.

**Example**

```
make parallel
mpirun -np 4 ./parallel -m 256 -n 256 -i 100000 -s 1000
```

**Compile and Run**

You can also execute both of the above commands together with default values with `make run`.

## Visualize
### Plots
`./plot_solution.sh -m [y_size] -n [x_size]`

Plots the program output using [gnuplot](http://gnuplot.sourceforge.net).

Alternatively, you can compile, run, and plot the solution with default values with `make plot` .

You can plot the sequential solution with `make plot_sequential`.

**Example**

`./plot_solution.sh -m 256 -n 256`

### Video
`make show`

Compiles, runs, and plots the solution with default values and creates a video using [ffmpeg](https://ffmpeg.org).

You can create a video from the sequential solution with `make show_sequential`.

## Check
`make check`

Compiles and runs the solution with default values and compares the output data to reference data.

You can check the sequential solution with `make check_sequential`.

## Options
Option | Description | Restrictions | Default value
:------------ | :------------ | :------------ | :------------
**-m** | Size of the y dimension | > 0 | 256
**-n** | Size of the x dimension | > 0 | 256
**-i** | Maximum number of iterations | > 0 | 100000
**-s** | Number of iterations between each time the state is saved to file | > 0 | 1000
**-np**| Number of processes (MPI option) | > 0 | 4

## Installing dependencies
**OpenMPI**

Linux/Ubuntu:

```
sudo apt update
sudo apt install -y openmpi-bin openmpi-doc libopenmpi-dev
```

MacOSX:

```
brew update
brew install open-mpi
```

**gnuplot**

Linux/Ubuntu:

```
sudo apt update
sudo apt install gnuplot
```

MacOSX:

```
brew update
brew install gnuplot
```

**ffmpeg**

Linux/Ubuntu:

```
sudo apt update
sudo apt install ffmpeg
```

MacOSX:

```
brew update
brew install ffmpeg
```

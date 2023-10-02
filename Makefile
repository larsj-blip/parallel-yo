PROGNAME:=heat

DATADIR:=data/
PLOTSDIR:=plots/
VIDEODIR:=video/

DATA:=$(DATADIR)*.bin
PLOTS:=$(PLOTSDIR)*.png
VIDEO:=$(VIDEODIR)*.mp4

CC:=gcc
PARALLEL_CC:=mpicc
CFLAGS+=-std=c11
LDLIBS+=-lm

PARALLEL_SRC_FILES:=src/$(PROGNAME)_parallel.c src/argument_utils.c
SERIAL_SRC_FILES:=src/$(PROGNAME)_sequential.c src/argument_utils.c
SRC_FILES:=$(PARALLEL_SRC_FILES) $(SERIAL_SRC_FILES)

.PHONY: clean purge setup run check plot show run_sequential run_parallel check_sequential check_parallel plot_sequential plot_parallel show_sequential show_parallel viz


parallel: $(PARALLEL_SRC_FILES)
	$(PARALLEL_CC) $^ $(CFLAGS) -o $@ $(LDLIBS)

sequential: $(SERIAL_SRC_FILES)
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

run: run_parallel
check: check_parallel
plot: plot_parallel
show: show_parallel

clean:
	-rm -f sequential parallel

purge:
	-rm -f sequential parallel $(DATA) $(PLOTS) $(VIDEO)

setup:
	-mkdir -p data plots video
	$(MAKE) -C check clean
	$(MAKE) -C check all

run_sequential: purge sequential
	./sequential

run_parallel: purge parallel
	mpirun -np 4 --oversubscribe ./parallel

check_sequential: purge sequential
	./check/check_sequential_solution.sh

check_parallel: purge parallel
	./check/check_parallel_solution.sh

plot_sequential: purge run_sequential
	./plot_results.sh

plot_parallel: purge run_parallel
	./plot_results.sh

show_sequential: purge run_sequential viz
show_parallel: purge run_parallel viz

viz:
	./plot_results.sh > /dev/null
	ffmpeg -y -i $(PLOTSDIR)%5d.png -vf format=yuv420p $(VIDEODIR)animation.mp4 &> /dev/null
	./open_video.sh &> /dev/null

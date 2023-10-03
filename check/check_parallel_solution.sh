echo
echo "--------------------------------------------------------------------------------"
echo "Comparing parallel solution with reference data"
echo "--------------------------------------------------------------------------------"
echo "M: 256, N: 256, max iteration: 100000, snapshot frequency: 10000"
echo "--------------------------------------------------------------------------------"
echo
for i in 1 2 4 8
do
    echo "Running with $i processes:"
    mpirun -n $i --oversubscribe ./parallel 1>/dev/null
    ./check/compare_solutions 256 256 data/00050.bin check/references/parallel/00050.bin
    echo
done

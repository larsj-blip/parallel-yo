help()
{
    echo
    echo "Plot 2D heat"
    echo
    echo "Syntax"
    echo "---------------------------------------------"
    echo "./plot_results.sh [-m|n|h]                    "
    echo
    echo "Option    Description     Arguments   Default"
    echo "---------------------------------------------"
    echo "m         x size          Optional    256    "
    echo "n         y size          Optional    256    "
    echo "h         Help            None               "
    echo
    echo "Example"
    echo "---------------------------------------------"
    echo "./plot_solution.sh -m 256 -n 256           "
    echo
}

#-----------------------------------------------------------------
set -e

M=256
N=256

while getopts ":m:n:h" opt; do
    case $opt in
        m)
            M=$OPTARG;;
        n)
            N=$OPTARG;;
        h)
            help
            exit;;
        \?)
            echo "Invalid option"
            help
            exit;;
    esac
done

#-----------------------------------------------------------------
SIZE_M=`echo $M | bc`
SIZE_N=`echo $N | bc`

# Go through every data/#####.bin file
for data in `find data/*.bin`
do
    # Convert the file name into 'image/#####.png'
    image=`echo $data | sed s/data/plots/ | sed s/\.bin/.png/`
    # Give an indicator that something is happening
    echo "Plotting '$image'"

    # Pipe the plotting script into gnuplot, using a background process
    #  so that each plot doesn't have to wait for the others
    cat <<END_OF_SCRIPT | gnuplot - &
set term png
set view map
set cbrange [0:60]
set output "$image"
plot "$data" binary array=${SIZE_M}x${SIZE_N} format="%lf" with image
END_OF_SCRIPT
done

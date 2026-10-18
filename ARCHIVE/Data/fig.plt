#!/Users/jay/gnuplot/bin/gnuplot
load 'inferno.pal'
set style line  1 lt 1 lc rgb '#0c0887' # blue
set style line  2 lt 1 lc rgb '#4b03a1' # purple-blue
set style line  3 lt 1 lc rgb '#7d03a8' # purple
set style line  4 lt 1 lc rgb '#a82296' # purple
set style line  5 lt 1 lc rgb '#cb4679' # magenta
set style line  6 lt 1 lc rgb '#e56b5d' # red
set style line  7 lt 1 lc rgb '#f89441' # orange
set style line  8 lt 1 lc rgb '#fdc328' # orange
set style line  9 lt 1 lc rgb '#f0f921' # yellow

set terminal postscript enhanced color 'Helvetica' 20
set size square
set output 'pes.eps'

#set border linewidth 1
#set key spacing 1.05
unset xtics
unset ytics
unset border
set xrange [-1:1]
plot 'gam_10.0_pes.txt' u 1:3 w l lc rgb '#fdc328' lw 6 notitle, \
'gam_10.0_pes.txt' u 1:4 w l lc rgb '#cb4679' lw 6 notitle 


#!/Users/jay/gnuplot/bin/gnuplot
set terminal postscript enhanced color 'Helvetica' 20
set palette defined ( 0 "blue", 0.5 "light-grey", 1 "red" )
#set palette model XYZ rgbformulae 7,5,15
set pm3d map
set size square
set xlabel 'Dissipation Energy (meV)'
set ylabel 'Coupling Energy (eV)'
set output 'rs.eps'
splot 'rabi_plot.out' u ($1*27.211*1e3):($2*27.211):($7*27.211) w pm3d notitle

set output 'dc.eps'
#unset ylabel
#unset ytics
splot 'dc_map.out' u ($1*27.211*1e3):($2*27.211):3 w pm3d notitle
#set xrange [1:4]
#f = 0.0529177
#set yrange [2.7:3.8]
#set xlabel 'Energy (eV)'
#set ylabel 'Donor-Acceptor 2 Separation (nm)'
#set output 'A1_A2_0.2nm.eps'
#splot 'compare_dsa2_0.2nm_separation_a1_a2.txt' u 2:1:3 notitle


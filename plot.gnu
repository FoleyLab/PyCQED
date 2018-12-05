#!/Users/jay/gnuplot/bin/gnuplot

set terminal postscript enhanced color 'Helvetica' 30
set output 'Lifetimes.eps'
set key bottom right
t = 0.02418
set xrange [0:20]
set xlabel 'Time (fs)'
set ylabel 'Population'
plot 'NormalLifetime.txt' u ($1*t):2 w l lw 5 lc rgb 'black' title '{/Symbol r}_g, Bare Molecule', \
'LifetimeEnhancement.txt' u ($1*t):2 w l lw 5 lc rgb 'red' title '{/Symbol r}_g, Molecule + High Q', \
'LifetimeDiminishment2.txt' u ($1*t):2 w l lw 5 lc rgb 'blue' title '{/Symbol r}_g, Molecule + Low Q', \

unset ylabel
unset ytics
set output 'Molecule_HighQ.eps'
set key top right
plot 'LifetimeEnhancement.txt' u ($1*t):($7) w l lw 5 lc rgb 'black' title '{/Symbol r}_{LP}', \
'LifetimeEnhancement.txt' u ($1*t):3 w l lw 5 lc rgb 'red' title '{/Symbol r}_{0e}', \
'LifetimeEnhancement.txt' u ($1*t):4 w l lw 5 lc rgb 'blue' title '{/Symbol r}_{1g}', \

set output 'Molecule_LowQ.eps'
set key top right
plot 'LifetimeDiminishment2.txt' u ($1*t):($7) w l lw 5 lc rgb 'black' title '{/Symbol r}_{LP}', \
'LifetimeDiminishment2.txt' u ($1*t):3 w l lw 5 lc rgb 'red' title '{/Symbol r}_{0e}', \
'LifetimeDiminishment2.txt' u ($1*t):4 w l lw 5 lc rgb 'blue' title '{/Symbol r}_{1g}', \



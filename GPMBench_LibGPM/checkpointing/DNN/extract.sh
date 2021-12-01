cd results
for f in ./* ; do
    printf '%s\t' $f
done
printf '\n'
for f in ./* ; do
    time=$(tac $f | grep -m1 'Overall Execution time' | grep -oE '[0-9]+\.{0,1}[0-9]*')
    printf '%f\t' $time
done
printf '\n'
cd ..

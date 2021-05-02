i = 0
for s in 1 2 3
do
 for i in $(seq 0 50)
 do
  echo "$s":"$i"
 done
done

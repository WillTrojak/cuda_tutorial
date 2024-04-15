
compute_capability=70
file="legendre.cu"
exec="go"

rm -f ${exec}

nvcc \
-o ${exec} \
-O3 \
--dopt on \
--use_fast_math \
--std=c++17 \
-arch sm_${compute_capability} \
${file}

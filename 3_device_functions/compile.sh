compute_capability=70
file="gauss_0.cu"
exec="go"

rm -f ${exec}

nvcc \
-o ${exec} \
-O3 \
--dopt on \
--use_fast_math \
--std=c++17 \
--ptxas-options="-v" \
-arch sm_${compute_capability} \
${file}

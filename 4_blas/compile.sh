
compute_capability=70
file="legendre.cu"
exec="go"

rm -f ${exec}

nvcc \
-arch compute_${compute_capability} \
-code sm_${compute_capability} \
-lcublas \
-o ${exec} ${file}

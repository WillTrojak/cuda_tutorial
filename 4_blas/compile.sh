
compute_capability=70
file="legendre.cu"
exec="go"

rm -f ${exec}

nvcc \
-o ${exec} \
-arch sm_${compute_capability} \
${file}

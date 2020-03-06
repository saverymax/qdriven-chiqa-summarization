for q in with_question without_question
do
    echo $q
    sbatch --job-name=$q --export=QDRIVEN=$q train_medsumm.sh
done

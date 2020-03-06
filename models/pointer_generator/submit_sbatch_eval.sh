for q in with_question without_question
do
    echo $q
    sbatch --job-name=${q}_eval --export=QDRIVEN=$q eval_medsumm.sh
done

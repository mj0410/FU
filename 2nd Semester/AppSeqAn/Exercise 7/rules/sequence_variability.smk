rule calculate_variability:
    input:
        "alignment/alignment.fasta"
    output:
        txt = "variability/avg_sequence_variability.txt",
        plot = "variability/plot_variability.png"
    conda:
        "../envs/Bio_env.yaml"
    script:
        "../scripts/variability.py"

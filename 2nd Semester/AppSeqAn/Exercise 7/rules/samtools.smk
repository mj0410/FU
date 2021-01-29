rule sam_to_bam:
    input:
        "sam/{sample}.sam"
    output:
        "bam/{sample}.bam"
    conda:
        "../envs/env.yaml"
    threads: 4
    shell:
        "samtools view --threads {threads} -S -b {input} > {output}"

rule sort_bam:
    input:
        "bam/{sample}.bam"
    output:
        "bam_sorted/{sample}.bam"
    conda:
        "../envs/env.yaml"
    threads: 4
    shell:
        "samtools sort --threads {threads} -o {output} {input}"

rule index_bam:
    input:
         "bam_sorted/{sample}.bam"
    output:
         "bam_sorted/{sample}.bam.bai"
    conda:
        "../envs/env.yaml"
    threads: 4
    shell:
         "samtools index --threads {threads} {input} {output}"

rule samtools_idxstats:
    input:
        "bam_sorted/{sample}.bam"
    output:
        "stats/{sample}_idxstats.txt"
    conda:
        "../envs/env.yaml"
    threads: 4
    shell:
        "samtools idxstats --threads {threads} {input} > {output}"

rule new_stats:
    input:
        a="stats/{sample}_idxstats.txt"
    output:
        b="new_stats/{sample}_idxstats_new.txt"
    conda:
        "../envs/env.yaml"
    script:
        "../scripts/new_stats.py"

rule index_build:
    input:
        config["index"]
    output:
        "reference/ref.1.bt2"
    conda:
        "../envs/env.yaml"
    shell:
        "bowtie2-build {input} reference/ref"

rule read_paired_end:
    input:
        fastq1 = "trimmed_reads/{sample}_1.fastq",
        fastq2 = "trimmed_reads/{sample}_2.fastq",
        ref_index = "reference/ref.1.bt2"
    output:
        "mapping/sam/{sample}.sam"
    conda:
        "../envs/env.yaml"
    threads:
        config['threads']
    shell:
        "bowtie2 -x reference/ref --threads {threads} -1 {input.fastq1} -2 {input.fastq2} -S {output}"

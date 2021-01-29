rule sam_to_bam:
    input:
        "mapping/sam/{sample}.sam"
    output:
        "mapping/bam/{sample}.bam"
    conda:
        "../envs/env.yaml"
    threads:
        config['threads']
    shell:
        "samtools view --threads {threads} -S -b {input} > {output}"

rule sort_bam:
    input:
        "mapping/bam/{sample}.bam"
    output:
        "mapping/bam_sorted/{sample}.bam"
    conda:
        "../envs/env.yaml"
    threads:
        config['threads']
    shell:
        "samtools sort --threads {threads} -o {output} {input}"

rule index_bam:
    input:
         "mapping/bam_sorted/{sample}.bam"
    output:
         "mapping/bam_sorted/{sample}.bam.bai"
    conda:
        "../envs/env.yaml"
    threads:
        config['threads']
    shell:
         "samtools index -@ {threads} {input} {output}"

rule mpileup:
    input:
         bam = "mapping/bam_sorted/{sample}.bam",
         ref = config['index']
    output:
         "mapping/calls/{sample}.vcf.gz"
    conda:
        "../envs/env.yaml"
    threads:
        config['threads']
    shell:
         """
         bcftools mpileup --threads {threads} -Ou -f {input.ref} {input.bam} | bcftools call --threads {threads} -m -Oz -o {output}
         bcftools index {output}
         """

rule samtools_idxstats:
    input:
        "mapping/bam_sorted/{sample}.bam"
    output:
        "mapping/stats/{sample}_idxstats.txt"
    conda:
        "../envs/env.yaml"
    threads:
        config['threads']
    shell:
        "samtools idxstats --threads {threads} {input} > {output}"

rule new_stats:
    input:
        a="mapping/stats/{sample}_idxstats.txt"
    output:
        b="mapping/new_stats/{sample}_idxstats_new.txt"
    conda:
        "../envs/env.yaml"
    script:
        "../scripts/new_stats.py"

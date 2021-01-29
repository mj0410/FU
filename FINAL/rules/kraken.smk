rule kraken:
    input:
        r1=lambda wildcards: samples.at[wildcards.sample, 'fq1'],
        r2=lambda wildcards: samples.at[wildcards.sample, 'fq2']
    output:
        output="kraken/{sample}.tsv",
        report="kraken/{sample}_report.tsv"
    conda:
        "../envs/env.yaml"
    params:
        db = config['database']
    threads:
        config['threads']
    shell:
        "kraken2 --threads {threads} --db {params.db} --paried {input.r1} {input.r2} --output {output.output} --report {output.report}"


rule bracken:
    input:
        "kraken/{sample}_report.tsv"
    output:
        "bracken/{sample}_bracken.tsv"
    conda:
        "../envs/env.yaml"
    params:
        db = config['database']
    shell:
        "bracken -d {params.db} -i {input} -o {output}"

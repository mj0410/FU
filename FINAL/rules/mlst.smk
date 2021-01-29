rule mlst:
    input:
        expand("contigs/{sample}.fa", sample=list(samples.index))
    output:
        "mlst/mlst.tsv"
    conda:
        "../envs/env.yaml"
    params:
        species = config['species']
    threads:
        config['threads']
    shell:
        "mlst --scheme {params.species} --threads {threads} {input} > {output}"

rule annotation:
    input:
        "contigs/{sample}.fa"
    output:
        "prokka/prokka_{sample}/{sample}.gff"
    params:
        outdir = "prokka/prokka_{sample}",
        genome = "{sample}"
    conda:
        "../envs/annotation.yaml"
    shell:
        "prokka --kingdom Bacteria --centre X --compliant --outdir {params.outdir} --prefix {params.genome} --force {input}"

rule core_genome:
    input:
        expand("prokka/prokka_{sample}/{sample}.gff", sample=list(samples.index))
    output:
        "roary/core_gene_alignment.aln"
    conda:
        "../envs/env.yaml"
    params:
        outdir = "roary_results",
        outfile = "roary_results/core_gene_alignment.aln"
    threads:
        config['threads']
    shell:
        """
        roary -f {params.outdir} -e -n -p {threads} {input}

        mv {params.outfile} {output}
        """

rule tree:
    input:
        "roary/core_gene_alignment.aln"
    output:
        "tree/tree"
    conda:
        "../envs/env.yaml"
    shell:
        "FastTree -gtr -nt -fastest {input} > {output}"

rule to_png:
    input:
        "tree/tree"
    output:
        "tree/phylogenetic_tree.png"
    conda:
        "../envs/tree_env.yaml"
    script:
        "../scripts/to_png.py"

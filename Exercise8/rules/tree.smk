rule tree:
    input:
        "alignment/alignment.fasta"
    output:
        "tree/tree"
    conda:
        "../envs/env.yaml"
    shell:
        "FastTree -gtr -nt {input} > {output}"

rule to_png:
    input:
        "tree/tree"
    output:
        "tree/phylogenetic_tree.png"
    conda:
        "../envs/tree_env.yaml"
    script:
        "../scripts/to_png.py"

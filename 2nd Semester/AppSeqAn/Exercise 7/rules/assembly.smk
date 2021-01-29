rule downsample:
    input:
        "trimmed_reads/{sample}_1.fastq", "trimmed_reads/{sample}_2.fastq"
    output:
        "normalized/{sample}_1.fastq",
        "normalized/{sample}_2.fastq"
    conda:
        "../envs/env.yaml"
    shell:
        "bbnorm.sh in1={input[0]} in2={input[1]} out1={output[0]} out2={output[1]} target=500"


rule velvet:
    input:
        "normalized/{sample}_1.fastq",
        "normalized/{sample}_2.fastq"
    output:
        "contigs/{sample}.fa"
    params:
        out = "velvet/contigs.fa"
    conda:
        "../envs/env.yaml"
    shell:
        """
        velveth velvet 80 -fastq -short -separate {input[0]} {input[1]}
        velvetg velvet

        mv {params.out} {output}
        """

rule scaffolding:
    input:
        ref = config['index'],
        contig = "contigs/{sample}.fa"
    output:
        "assembly/{sample}.fasta"
    params:
        "abacas/abacas.fasta"
    conda:
        "../envs/env.yaml"
    shell:
        """
        mkdir -p abacas
        abacas.pl -r {input.ref} -q {input.contig} -p nucmer -o abacas/abacas

        mv {params} {output}

        mv -t abacas nucmer.delta nucmer.filtered.delta nucmer.tiling unused_contigs.out
        """

rule edit_header:
    input:
        "assembly/{sample}.fasta"
    output:
        "assembly_edit/{sample}.fasta"
    conda:
        "../envs/Bio_env.yaml"
    script:
        "../scripts/rename.py"


rule alignment:
    input:
        sequence = expand("assembly_edit/{sample}.fasta", sample=list(samples.index)),
        reference = config['index']
    output:
        "alignment/alignment.fasta"
    conda:
        "../envs/env.yaml"
    shell:
        "augur align -s {input.sequence} --reference-sequence {input.reference} -o {output}"

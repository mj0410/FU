rule downsample:
    input:
        "trimmed_reads/{sample}_1.fastq", "trimmed_reads/{sample}_2.fastq"
    output:
        "normalized/{sample}_1.fastq",
        "normalized/{sample}_2.fastq"
    conda:
        "../envs/env.yaml"
    shell:
        "bbnorm.sh in1={input[0]} in2={input[1]} out1={output[0]} out2={output[1]} target=10"

if config["assembly"] == "denovo":
    rule velvet:
        input:
            "normalized/{sample}_1.fastq",
            "normalized/{sample}_2.fastq"
        output:
            "contigs/{sample}.fa"
        params:
            out = "velvet/velvet_{sample}/contigs.fa",
            outdir = "velvet/velvet_{sample}"
        conda:
            "../envs/env.yaml"
        shell:
            """
            mkdir -p {params.outdir}

            velveth {params.outdir} 89 -fastq -separate {input[0]} {input[1]}
            velvetg {params.outdir}

            mv {params.out} {output}
            """

    rule scaffolding:
        input:
            ref = config['index'],
            contig = "contigs/{sample}.fa"
        output:
            "abacas/abacas_{sample}/{sample}.fasta"
        params:
            outdir = "abacas_{sample}",
            prefix = "{sample}"
        conda:
            "../envs/env.yaml"
        shell:
            """
            cwd=$(pwd)
            refdir=$(pwd)/{input.ref}
            datadir=$(pwd)/{input.contig}

            mkdir -p abacas
            cd abacas
            mkdir -p {params.outdir}
            cd {params.outdir}

            abacas.pl -r $refdir -q $datadir -p nucmer -o {params.prefix}

            cd $cwd
            """

    rule edit_header:
        input:
            "abacas/abacas_{sample}/{sample}.fasta"
        output:
            "assembly/{sample}.fasta"
        conda:
            "../envs/Bio_env.yaml"
        script:
            "../scripts/rename.py"

else:
    rule consensus:
        input:
            ref = config['index'],
            vcf = "mapping/calls/{sample}.vcf.gz"
        output:
            "contigs/{sample}.fa"
        params:
            prefix = "contigs/{sample}"
        conda:
            "../envs/env.yaml"
        shell:
            "cat {input.ref} | bcftools consensus {input.vcf} > {output}"

    rule move_to_assembly:
        input:
            "contigs/{sample}.fa"
        output:
            "assembly/{sample}.fasta"
        conda:
            "../envs/Bio_env.yaml"
        script:
            "../scripts/rename.py"

rule alignment:
    input:
        sequence = expand("assembly/{sample}.fasta", sample=list(samples.index)),
        reference = config['index']
    output:
        "alignment/alignment.fasta"
    conda:
        "../envs/env.yaml"
    shell:
        "augur align -s {input.sequence} --reference-sequence {input.reference} -o {output}"

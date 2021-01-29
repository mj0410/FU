rule fastqc1:
    input:
        lambda wildcards: samples.at[wildcards.sample, 'fq1']
    output:
        "qc_report/reports/{sample}_1_fastqc.zip",
        "qc_report/reports/{sample}_1_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_path = lambda wildcards: samples.at[wildcards.sample, 'path'],
        fastqc_zip = "{sample}_1_fastqc.zip",
        fastqc_html = "{sample}_1_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_path}{params.fastqc_zip} {output[0]}
        mv {params.fastqc_path}{params.fastqc_html} {output[1]}
        """

rule fastqc2:
    input:
        lambda wildcards: samples.at[wildcards.sample, 'fq2']
    output:
        "qc_report/reports/{sample}_2_fastqc.zip",
        "qc_report/reports/{sample}_2_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_path = lambda wildcards: samples.at[wildcards.sample, 'path'],
        fastqc_zip = "{sample}_2_fastqc.zip",
        fastqc_html = "{sample}_2_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_path}{params.fastqc_zip} {output[0]}
        mv {params.fastqc_path}{params.fastqc_html} {output[1]}
        """

rule trimming:
    input:
        lambda wildcards: samples.at[wildcards.sample, 'fq1'],
        lambda wildcards: samples.at[wildcards.sample, 'fq2']
    output:
        "trimmed_reads/{sample}_1.fastq",
        "trimmed_reads/unpaired_{sample}_1.fastq",
        "trimmed_reads/{sample}_2.fastq",
        "trimmed_reads/unpaired_{sample}_2.fastq"
    conda:
        "../envs/env.yaml"
    shell:
        "trimmomatic PE {input} {output} LEADING:2 TRAILING:2 SLIDINGWINDOW:4:15 ILLUMINACLIP:TruSeq3-PE.fa:2:30:10:2:true"

rule fastqc_filter:
    input:
        "trimmed_reads/{sample}.fastq"
    output:
        "qc_report/trimmed_reports/{sample}_fastqc.zip",
        "qc_report/trimmed_reports/{sample}_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_zip = "trimmed_reads/{sample}_fastqc.zip",
        fastqc_html = "trimmed_reads/{sample}_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_zip} {output[0]}
        mv {params.fastqc_html} {output[1]}
        """

rule qualimap:
    input:
        "mapping/bam_sorted/{sample}.bam"
    output:
        "qc_report/qualimap/{sample}.pdf"
    conda:
        "../envs/env.yaml"
    params:
        pdf_path = "mapping/bam_sorted/{sample}_stats/report.pdf"
    shell:
        """
        qualimap bamqc -bam {input} -outformat pdf

        mv {params.pdf_path} {output}
        """


if config['kraken']=='true':#execute rule kraken and load the result
    rule multiqc:
       input:
            expand("qc_report/reports/{sample}_1_fastqc.html", sample=list(samples.index)),
            expand("qc_report/reports/{sample}_2_fastqc.html", sample=list(samples.index)),
            expand("qc_report/trimmed_reports/{sample}_fastqc.html", sample=list(samples.index+"_1")),
            expand("qc_report/trimmed_reports/{sample}_fastqc.html", sample=list(samples.index+"_2")),
            expand("qc_report/qualimap/{sample}.pdf", sample=list(samples.index)),
            expand("bracken/{sample}_bracken.tsv", sample=list(samples.index))
       output:
           "qc_report/multiqc_report.html"
       conda:
            "../envs/env.yaml"
       shell:
            "multiqc qc_report bracken -o qc_report"

else:
    rule multiqc:
       input:
            expand("qc_report/reports/{sample}_1_fastqc.html", sample=list(samples.index)),
            expand("qc_report/reports/{sample}_2_fastqc.html", sample=list(samples.index)),
            expand("qc_report/trimmed_reports/{sample}_fastqc.html", sample=list(samples.index+"_1")),
            expand("qc_report/trimmed_reports/{sample}_fastqc.html", sample=list(samples.index+"_2")),
            expand("qc_report/qualimap/{sample}.pdf", sample=list(samples.index))
       output:
           "qc_report/multiqc_report.html"
       conda:
            "../envs/env.yaml"
       shell:
            "multiqc qc_report -o qc_report"

#conda:
#   "envs/env.yaml"

rule fastqc_1:
    input:
        lambda wildcards: samples.at[wildcards.sample, 'fq1']
    output:
        "reports/{sample}_1_fastqc.zip",
        "reports/{sample}_1_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_zip = lambda wildcards: samples.at[wildcards.sample, 'fq1'].rpartition('.')[0] + "_fastqc.zip",
        fastqc_html = lambda wildcards: samples.at[wildcards.sample, 'fq1'].rpartition('.')[0] + "_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_zip} {output[0]}
        mv {params.fastqc_html} {output[1]}
        """

rule fastqc_2:
    input:
        lambda wildcards: samples.at[wildcards.sample, 'fq2']
    output:
        "reports/{sample}_2_fastqc.zip",
        "reports/{sample}_2_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_zip = lambda wildcards: samples.at[wildcards.sample, 'fq2'].rpartition('.')[0] + "_fastqc.zip",
        fastqc_html = lambda wildcards: samples.at[wildcards.sample, 'fq2'].rpartition('.')[0] + "_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_zip} {output[0]}
        mv {params.fastqc_html} {output[1]}
        """

rule trimming:
    input:
        lambda wildcards: samples.at[wildcards.sample, 'fq1'],
        lambda wildcards: samples.at[wildcards.sample, 'fq2']
    output:
        "trimmed_reads/paired_{sample}_1.fastq",
        "trimmed_reads/uhnpaired_{sample}_1.fastq",
        "trimmed_reads/paired_{sample}_2.fastq",
        "trimmed_reads/unpaired_{sample}_2.fastq"
    conda:
        "../envs/env.yaml"
    shell:
        "trimmomatic PE {input} {output} LEADING:2 TRAILING:2 SLIDINGWINDOW:4:15 MINLEN:25"

rule fastqc_filter:
    input:
        "trimmed_reads/paired_{sample}.fastq"
    output:
        "trimmed_reports/{sample}_fastqc.zip",
        "trimmed_reports/{sample}_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_zip = "trimmed_reads/paired_{sample}_fastqc.zip",
        fastqc_html = "trimmed_reads/paired_{sample}_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_zip} {output[0]}
        mv {params.fastqc_html} {output[1]}
        """

rule qualimap:
    input:
        "bam_sorted/{sample}.bam"
    output:
        "reports/qualimap/{sample}.pdf"
    conda:
        "../envs/env.yaml"
    params:
        pdf_path = "bam_sorted/{sample}_stats/report.pdf"
    shell:
        """
        qualimap bamqc -bam {input} -outformat pdf

        mv {params.pdf_path} {output}
        """

rule multiqc:
   input:
       expand("trimmed_reports/{sample}_1_fastqc.html", sample=list(samples.index)),
       expand("trimmed_reports/{sample}_2_fastqc.html", sample=list(samples.index)),
       expand("reports/qualimap/{sample}.pdf", sample=list(samples.index))
   output:
       "reports/multiqc_report.html"
   conda:
       "../envs/env.yaml"
   shell:
       "multiqc trimmed_reports -o reports"

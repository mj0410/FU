rule fastqc_1:
    input:
        config["data"]+"{sample}_1.fastq"
    output:
        "reports/{sample}_1_fastqc.zip",
        "reports/{sample}_1_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_zip = config["data"]+"{sample}_1_fastqc.zip",
        fastqc_html = config["data"]+"{sample}_1_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_zip} {output[0]}
        mv {params.fastqc_html} {output[1]}
        """

rule fastqc_2:
    input:
        config["data"]+"{sample}_2.fastq"
    output:
        "reports/{sample}_2_fastqc.zip",
        "reports/{sample}_2_fastqc.html"
    conda:
        "../envs/env.yaml"
    params:
        fastqc_zip = config["data"]+"{sample}_2_fastqc.zip",
        fastqc_html = config["data"]+"{sample}_2_fastqc.html"
    shell:
        """
        fastqc {input}

        mv {params.fastqc_zip} {output[0]}
        mv {params.fastqc_html} {output[1]}
        """

rule trimmomatic_pe:
    input:
        r1=config["data"]+"{sample}_1.fastq",
        r2=config["data"]+"{sample}_2.fastq"
    output:
        r1="trimmed_reads/{sample}_1.fastq",
        r2="trimmed_reads/{sample}_2.fastq",
        # reads where trimming entirely removed the mate
        r1_unpaired="trimmed_reads/{sample}_1.unpaired.fastq",
        r2_unpaired="trimmed_reads/{sample}_2.unpaired.fastq"
    log:
        "logs/trimmomatic/{sample}.log"
    params:
        # list of trimmers (see manual)
        trimmer=["TRAILING:3"],
        # optional parameters
        extra="",
        compression_level="-9"
    threads:
        4
    wrapper:
        "0.60.0/bio/trimmomatic/pe"

rule fastqc_filter:
    input:
        "trimmed_reads/{sample}.fastq"
    output:
        "trimmed_reports/{sample}_fastqc.zip",
        "trimmed_reports/{sample}_fastqc.html"
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

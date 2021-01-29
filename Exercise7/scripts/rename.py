from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

file_in = str(snakemake.input)
file_out = str(snakemake.output)

start = file_in.find('/')+1
end = file_in.find('.', start)

id = file_in[start:end]

print(id)

with open(file_out, 'w') as f_out:
    for seq_record in SeqIO.parse(open(file_in, mode='r'), 'fasta'):
        # remove .id from .description record (remove all before first space)
        seq_record.description=' '.join(seq_record.description.split()[1:])
        seq_record.id = id
        r=SeqIO.write(seq_record, f_out, 'fasta')
        if r!=1: print('Error while writing sequence:  ' + seq_record.id)

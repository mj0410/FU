#!/bin/sh
input_sequences="sequences.fasta"
input_metadata="sequences.csv"
input_directory="final_results"
input_reference="COVID_19_reference.gb"

params_coalescent="opt"
date_inference="marginal"
clock_filter_iqd=4
inference="joint"
columns="region	country"
colors="colors.csv"
lat_longs="lat_longs.tsv"
auspice_config="auspice_config.json"
"
echo "Filter: $input_directory/$input_sequences "
augur filter \
            --sequences $input_directory/$input_sequences \
            --metadata $input_directory/$input_metadata \
            --output "$input_directory/filtered.fasta" \

echo "Align: $input_directory/filtered.fasta"
augur align \
            --sequences "$input_directory/filtered.fasta" \
            --reference-sequence "$input_directory/$input_reference" \
            --output "$input_directory/aligned.filtered.fasta" \
            --fill-gaps           

echo "Tree: $input_directory/aligned.filtered.fasta"
augur tree \
            --alignment "$input_directory/aligned.filtered.fasta" \
            --output "$input_directory/output.tree"
    
echo "Refine: $input_directory/aligned.filtered.fasta"
augur refine \
            --tree "$input_directory/output.tree" \
            --alignment "$input_directory/aligned.filtered.fasta" \
            --metadata $input_directory/$input_metadata \
            --output-tree "$input_directory/output.refine.tree" \
            --output-node-data "$input_directory/output.node_data" \
            --timetree \
            --coalescent $params_coalescent \
            --date-confidence \
            --date-inference $date_inference \
            --clock-filter-iqd $clock_filter_iqd

echo "Ancestral: $input_directory/aligned.filtered.fasta"
augur ancestral \
            --tree "$input_directory/output.refine.tree" \
            --alignment "$input_directory/aligned.filtered.fasta" \
            --output-node-data "$input_directory/nt_muts.json" \
            --inference $inference

echo "Translate: $input_directory/aligned.filtered.fasta"
augur translate \
            --tree "$input_directory/output.refine.tree" \
            --ancestral-sequences "$input_directory/nt_muts.json" \
            --reference-sequence "$input_directory/$input_reference" \
            --output-node-data "$input_directory/aa_muts.json" 

echo "Taits: $input_directory/aligned.filtered.fasta"
augur traits \
            --tree "$input_directory/output.refine.tree"  \
            --metadata $input_directory/$input_metadata \
            --output-node-data "$input_directory/traits.json" \
            --columns $columns \
            --confidence
"
echo "Export: $input_directory/aligned.filtered.fasta"
augur export v2 \
            --tree "$input_directory/output.refine.tree"  \
            --metadata $input_directory/$input_metadata \
            --node-data "$input_directory/output.node_data" "$input_directory/traits.json" "$input_directory/nt_muts.json" "$input_directory/aa_muts.json" \
            --colors "$input_directory/$colors" \
            --lat-longs $input_directory/$lat_longs \
            --auspice-config $input_directory/$auspice_config \
            --output "$input_directory/auspice_json"
       
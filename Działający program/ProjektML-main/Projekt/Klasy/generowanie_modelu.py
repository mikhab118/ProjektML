import subprocess

# Dane dot, które chcesz zapisać do pliku .dot
dot_data = """
digraph {
    graph [size="17.55,17.55"]
    node [align=left fontname=monospace fontsize=10 height=0.2 ranksep=0.1 shape=box style=filled]
    2417481460784 [label="
     (1, 3)" fillcolor=darkolivegreen1]
    2417481307792 [label=AddBackward0]
    2417481312496 -> 2417481307792
    2417481312496 [label=AddmmBackward0]
    2417481309808 -> 2417481312496
    2417326244208 [label="value_stream.2.bias
     (1)" fillcolor=lightblue]
    2417326244208 -> 2417481309808
    2417481309808 [label=AccumulateGrad]
    2417481315520 -> 2417481312496
    2417481315520 [label=ReluBackward0]
    2417481315760 -> 2417481315520
    2417481315760 [label=AddmmBackward0]
    2417481313216 -> 2417481315760
    2417326244016 [label="value_stream.0.bias
     (25)" fillcolor=lightblue]
    2417326244016 -> 2417481313216
    2417481313216 [label=AccumulateGrad]
    2417481312304 -> 2417481315760
    2417481312304 [label=SliceBackward0]
    2417481313264 -> 2417481312304
    2417481313264 [label=SelectBackward0]
    2417481312112 -> 2417481313264
    2417481312112 [label=SliceBackward0]
    2417481312208 -> 2417481312112
    2417481312208 [label=CudnnRnnBackward0]
    2417481315424 -> 2417481312208
    2417325948176 [label="lstm.weight_ih_l0
     (200, 8)" fillcolor=lightblue]
    2417325948176 -> 2417481315424
    2417481315424 [label=AccumulateGrad]
    2417481315376 -> 2417481312208
    2417325947408 [label="lstm.weight_hh_l0
     (200, 50)" fillcolor=lightblue]
    2417325947408 -> 2417481315376
    2417481315376 [label=AccumulateGrad]
    2417481313360 -> 2417481312208
    2417325947696 [label="lstm.bias_ih_l0
     (200)" fillcolor=lightblue]
    2417325947696 -> 2417481313360
    2417481313360 [label=AccumulateGrad]
    2417481315472 -> 2417481312208
    2417325945296 [label="lstm.bias_hh_l0
     (200)" fillcolor=lightblue]
    2417325945296 -> 2417481315472
    2417481315472 [label=AccumulateGrad]
    2417481315184 -> 2417481312208
    2417325947600 [label="lstm.weight_ih_l1
     (200, 50)" fillcolor=lightblue]
    2417325947600 -> 2417481315184
    2417481315184 [label=AccumulateGrad]
    2417481311440 -> 2417481312208
    2417325945392 [label="lstm.weight_hh_l1
     (200, 50)" fillcolor=lightblue]
    2417325945392 -> 2417481311440
    2417481311440 [label=AccumulateGrad]
    2417481311488 -> 2417481312208
    2417325948080 [label="lstm.bias_ih_l1
     (200)" fillcolor=lightblue]
    2417325948080 -> 2417481311488
    2417481311488 [label=AccumulateGrad]
    2417481311248 -> 2417481312208
    2417325946640 [label="lstm.bias_hh_l1
     (200)" fillcolor=lightblue]
    2417325946640 -> 2417481311248
    2417481311248 [label=AccumulateGrad]
    2417481315088 -> 2417481312208
    2417325948656 [label="lstm.weight_ih_l2
     (200, 50)" fillcolor=lightblue]
    2417325948656 -> 2417481315088
    2417481315088 [label=AccumulateGrad]
    2417481310672 -> 2417481312208
    2417325948560 [label="lstm.weight_hh_l2
     (200, 50)" fillcolor=lightblue]
    2417325948560 -> 2417481310672
    2417481310672 [label=AccumulateGrad]
    2417481310720 -> 2417481312208
    2417325947792 [label="lstm.bias_ih_l2
     (200)" fillcolor=lightblue]
    2417325947792 -> 2417481310720
    2417481310720 [label=AccumulateGrad]
    2417481310768 -> 2417481312208
    2417325948848 [label="lstm.bias_hh_l2
     (200)" fillcolor=lightblue]
    2417325948848 -> 2417481310768
    2417481310768 [label=AccumulateGrad]
    2417481312256 -> 2417481315760
    2417481312256 [label=TBackward0]
    2417481312160 -> 2417481312256
    2417326243920 [label="value_stream.0.weight
     (25, 50)" fillcolor=lightblue]
    2417326243920 -> 2417481312160
    2417481312160 [label=AccumulateGrad]
    2417481312352 -> 2417481312496
    2417481312352 [label=TBackward0]
    2417481312064 -> 2417481312352
    2417326244112 [label="value_stream.2.weight
     (1, 25)" fillcolor=lightblue]
    2417326244112 -> 2417481312064
    2417481312064 [label=AccumulateGrad]
    2417481308032 -> 2417481307792
    2417481308032 [label=SubBackward0]
    2417481310816 -> 2417481308032
    2417481310816 [label=AddmmBackward0]
    2417481313312 -> 2417481310816
    2417326244592 [label="advantage_stream.2.bias
     (3)" fillcolor=lightblue]
    2417326244592 -> 2417481313312
    2417481313312 [label=AccumulateGrad]
    2417481311632 -> 2417481310816
    2417481311632 [label=ReluBackward0]
    2417481310960 -> 2417481311632
    2417481310960 [label=AddmmBackward0]
    2417481313456 -> 2417481310960
    2417326244400 [label="advantage_stream.0.bias
     (25)" fillcolor=lightblue]
    2417326244400 -> 2417481313456
    2417481313456 [label=AccumulateGrad]
    2417481312304 -> 2417481310960
    2417481313408 -> 2417481310960
    2417481313408 [label=TBackward0]
    2417481313504 -> 2417481313408
    2417326244304 [label="advantage_stream.0.weight
     (25, 50)" fillcolor=lightblue]
    2417326244304 -> 2417481313504
    2417481313504 [label=AccumulateGrad]
    2417481310864 -> 2417481310816
    2417481310864 [label=TBackward0]
     2417481313552 -> 2417481310864
    2417326244496 [label="advantage_stream.2.weight
     (3, 25)" fillcolor=lightblue]
    2417326244496 -> 2417481313552
    2417481313552 [label=AccumulateGrad]
    2417481313120 -> 2417481308032
    2417481313120 [label=MeanBackward1]
    2417481310816 -> 2417481313120
    2417481307792 -> 2417481460784
}
"""

# Zapisz dane do pliku .dot
dot_filepath = "network_visualization.dot"
with open(dot_filepath, "w") as file:
    file.write(dot_data)

# Przekształć plik .dot w obraz .png za pomocą Graphviz
output_filepath = "network_visualization.png"
try:
    subprocess.run(["dot", "-Tpng", dot_filepath, "-o", output_filepath], check=True)
    print(f"Obraz zapisany jako {output_filepath}")
except Exception as e:
    print(f"Nie udało się przekształcić pliku .dot w .png: {e}")

## MLP 64-32-16-10, sklearn digits (360 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 20,621 | 21,002 | 210 | 5,816 / 5,848 | 152 | 96.11% | yes |
| tinymind int8 (+SMLAD) | 10,167 | 10,227 | 102 | 5,916 / 5,896 | 124 | 96.11% | yes |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 | 36,736 / 40,624 | 2,600 | 96.11% | yes |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 | 25,664 / 29,552 | 2,152 | 96.11% | yes |
| TFLM float32 | 27,902 | 28,059 | 281 | 43,228 / 47,276 | 1,824 | 96.11% | n/a (float) |
| emlearn float32 | 40,527 | 40,529 | 405 | 14,392 / 14,736 | 892 | 96.11% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| fc0 | 15,277 | 15,591 | 7,194 | 7,342 | 8,592 | 8,556 |
| fc1 | 4,214 | 4,169 | 2,166 | 2,225 | 3,421 | 3,384 |
| fc2 | 1,308 | 1,288 | 946 | 965 | 2,076 | 2,039 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 271,043 | 254,766 | 2,548 | 5,792 / 6,608 | 5,308 | yes |
| tinymind int8 (+SMLAD) | 195,841 | 181,853 | 1,819 | 5,408 / 6,252 | 5,372 | yes |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5,764 | 84,488 / 91,640 | 6,288 | yes |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24,356 | 41,088 / 46,192 | 5,668 | yes |
| TFLM float32 | 2,136,067 | 2,159,717 | 21,597 | 84,324 / 91,604 | 15,128 | n/a (float) |

### Per-layer cycles (kws, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv3x3 | 187,588 | 183,748 | 134,793 | 126,262 | 434,553 | 435,387 |
| maxpool | 10,166 | 10,158 | 10,161 | 10,156 | 35,807 | 36,383 |
| dwconv3x3 | 20,668 | 20,668 | 20,670 | 20,660 | 39,208 | 40,909 |
| pwconv | 45,709 | 41,835 | 29,877 | 27,485 | 53,212 | 53,863 |
| avgpool | 2,264 | 2,282 | 2,269 | 2,281 | 8,875 | 7,921 |
| fc | 1,312 | 1,302 | 954 | 952 | 2,097 | 2,031 |


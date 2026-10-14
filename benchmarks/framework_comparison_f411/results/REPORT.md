## MLP 64-32-16-10, sklearn digits (360 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 20,639 | 20,486 | 205 | 5,296 / 5,464 | 152 | 96.11% | yes |
| tinymind int8 (+SMLAD) | 10,305 | 10,241 | 102 | 5,380 / 5,548 | 124 | 96.11% | yes |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 | 36,736 / 40,624 | 2,600 | 96.11% | yes |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 | 25,664 / 29,552 | 2,152 | 96.11% | yes |
| TFLM float32 | 27,902 | 28,059 | 281 | 43,228 / 47,276 | 1,824 | 96.11% | n/a (float) |
| emlearn float32 | 40,527 | 40,529 | 405 | 14,392 / 14,736 | 892 | 96.11% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| fc0 | 15,096 | 15,097 | 7,344 | 7,345 | 8,592 | 8,556 |
| fc1 | 4,147 | 4,151 | 2,221 | 2,225 | 3,421 | 3,375 |
| fc2 | 1,279 | 1,280 | 948 | 949 | 2,076 | 2,035 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 266,451 | 254,900 | 2,549 | 4,996 / 6,848 | 5,332 | yes |
| tinymind int8 (+SMLAD) | 192,965 | 181,568 | 1,816 | 4,692 / 6,364 | 5,296 | yes |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5,764 | 84,488 / 91,640 | 6,288 | yes |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24,356 | 41,088 / 46,192 | 5,668 | yes |
| TFLM float32 | 2,136,067 | 2,159,717 | 21,597 | 84,324 / 91,604 | 15,128 | n/a (float) |

### Per-layer cycles (kws, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv3x3 | 181,968 | 173,164 | 132,155 | 127,756 | 434,553 | 435,412 |
| maxpool | 10,166 | 10,156 | 10,162 | 10,160 | 35,807 | 36,697 |
| dwconv3x3 | 21,413 | 21,054 | 21,415 | 21,047 | 39,208 | 40,918 |
| pwconv | 48,096 | 44,258 | 22,941 | 25,865 | 53,212 | 53,866 |
| avgpool | 2,266 | 2,282 | 2,267 | 2,281 | 8,875 | 7,917 |
| fc | 1,284 | 1,281 | 948 | 946 | 2,097 | 2,041 |


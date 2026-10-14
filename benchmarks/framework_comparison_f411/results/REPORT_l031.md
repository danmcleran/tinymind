## MLP 64-32-16-10, sklearn digits (32 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @32MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 35,637 | 35,963 | 1,124 | 5,488 / 5,784 | 200 | 96.88% | yes |
| TFLM int8 (CMSIS-NN) | does not fit | does not fit | - | 39,160 / 43,444 | 1,872+ | - | - |
| TFLM int8 (reference) | does not fit | does not fit | - | 33,944 / 38,292 | 1,776+ | - | - |
| TFLM float32 | does not fit | does not fit | - | 45,700 / 49,984 | 1,392+ | - | - |
| emlearn float32 | 712,272 | 707,113 | 22,097 | 18,720 / 18,988 | 972 | 96.88% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 |
|---|---:|---:|
| fc0 | 25,625 | 25,625 |
| fc1 | 7,707 | 7,707 |
| fc2 | 2,830 | 2,830 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @32MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 818,262 | 746,112 | 23,316 | 6,016 / 8,452 | 5,488 | yes |
| TFLM int8 (CMSIS-NN) | does not fit | does not fit | - | 83,000 / 90,468 | 5,136+ | - |
| TFLM int8 (reference) | does not fit | does not fit | - | 51,072 / 57,052 | 5,072+ | - |
| TFLM float32 | does not fit | does not fit | - | 82,852 / 90,304 | 14,448+ | - |

### Per-layer cycles (kws, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 |
|---|---:|---:|
| conv3x3 | 575,852 | 516,155 |
| maxpool | 24,921 | 24,921 |
| dwconv3x3 | 70,997 | 73,079 |
| pwconv | 137,957 | 137,483 |
| avgpool | 5,854 | 5,756 |
| fc | 2,830 | 2,830 |


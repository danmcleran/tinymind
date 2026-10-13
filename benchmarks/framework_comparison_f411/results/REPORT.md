## MLP 64-32-16-10, sklearn digits (360 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 21,478 | 21,481 | 215 | 5,128 / 5,296 | 148 | 96.11% | yes |
| TinyMind int8 (+SMLAD) | 11,093 | 11,160 | 112 | 5,196 / 5,364 | 124 | 96.11% | yes |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 | 36,736 / 40,624 | 2,600 | 96.11% | yes |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 | 25,664 / 29,552 | 2,152 | 96.11% | yes |
| TFLM float32 | 27,902 | 28,059 | 281 | 43,228 / 47,276 | 1,824 | 96.11% | n/a (float) |
| emlearn float32 | 40,527 | 40,529 | 405 | 14,392 / 14,736 | 892 | 96.11% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| fc0 | 15,537 | 15,672 | 7,698 | 7,724 | 8,592 | 8,556 |
| fc1 | 4,367 | 4,349 | 2,474 | 2,473 | 3,421 | 3,375 |
| fc2 | 1,422 | 1,423 | 1,127 | 1,127 | 2,076 | 2,035 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 411,419 | 381,463 | 3,815 | 3,248 / 4,988 | 5,088 | yes |
| TinyMind int8 (+SMLAD) | 352,124 | 329,792 | 3,298 | 3,060 / 4,948 | 5,064 | yes |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5,764 | 84,488 / 91,640 | 6,288 | yes |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24,356 | 41,088 / 46,192 | 5,668 | yes |
| TFLM float32 | 2,136,067 | 2,159,717 | 21,597 | 84,324 / 91,604 | 15,128 | n/a (float) |

### Per-layer cycles (kws, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv3x3 | 283,199 | 279,608 | 234,893 | 259,857 | 434,553 | 435,412 |
| maxpool | 14,480 | 10,157 | 14,480 | 10,161 | 35,807 | 36,697 |
| dwconv3x3 | 30,820 | 30,678 | 30,802 | 30,683 | 39,208 | 40,918 |
| pwconv | 65,712 | 55,519 | 51,217 | 44,169 | 53,212 | 53,866 |
| avgpool | 7,220 | 2,281 | 7,220 | 2,281 | 8,875 | 7,917 |
| fc | 1,440 | 1,419 | 1,125 | 1,131 | 2,097 | 2,041 |


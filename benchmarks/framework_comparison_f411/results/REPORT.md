## MLP 64-32-16-10, sklearn digits (360 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 20,639 | 20,486 | 205 | 5,296 / 5,464 | 152 | 96.11% | yes |
| TinyMind int8 (+SMLAD) | 10,305 | 10,241 | 102 | 5,380 / 5,548 | 124 | 96.11% | yes |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 | 36,736 / 40,624 | 2,600 | 96.11% | yes |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 | 25,664 / 29,552 | 2,152 | 96.11% | yes |
| TFLM float32 | 27,902 | 28,059 | 281 | 43,228 / 47,276 | 1,824 | 96.11% | n/a (float) |
| emlearn float32 | 40,527 | 40,529 | 405 | 14,392 / 14,736 | 892 | 96.11% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| fc0 | 15,096 | 15,097 | 7,344 | 7,345 | 8,592 | 8,556 |
| fc1 | 4,147 | 4,151 | 2,221 | 2,225 | 3,421 | 3,375 |
| fc2 | 1,279 | 1,280 | 948 | 949 | 2,076 | 2,035 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 329,733 | 272,639 | 2,726 | 4,076 / 5,744 | 5,164 | yes |
| TinyMind int8 (+SMLAD) | 276,387 | 229,438 | 2,294 | 3,916 / 5,792 | 5,164 | yes |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5,764 | 84,488 / 91,640 | 6,288 | yes |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24,356 | 41,088 / 46,192 | 5,668 | yes |
| TFLM float32 | 2,136,067 | 2,159,717 | 21,597 | 84,324 / 91,604 | 15,128 | n/a (float) |

### Per-layer cycles (kws, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv3x3 | 226,317 | 192,087 | 189,920 | 157,571 | 434,553 | 435,412 |
| maxpool | 10,162 | 10,157 | 10,162 | 10,154 | 35,807 | 36,697 |
| dwconv3x3 | 26,038 | 23,318 | 26,044 | 23,322 | 39,208 | 40,918 |
| pwconv | 60,753 | 47,553 | 41,121 | 38,659 | 53,212 | 53,866 |
| avgpool | 2,267 | 2,281 | 2,269 | 2,281 | 8,875 | 7,917 |
| fc | 1,270 | 1,286 | 942 | 956 | 2,097 | 2,041 |


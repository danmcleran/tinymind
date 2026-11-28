## MLPerf(R) Tiny reference keyword spotting model (DS-CNN, 49x10x1 -> 12), 8 random int8 inputs

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 16,671,846 | 15,677,563 | 156,776 | 37,892 / 53,072 | 17,288 | yes |
| tinymind int8 (+SMLAD) | 9,002,425 | 7,970,167 | 79,702 | 40,864 / 53,416 | 17,268 | yes |
| TFLM int8 (CMSIS-NN) | 9,692,160 | 9,583,099 | 95,831 | 138,928 / 148,864 | 24,432 | yes |
| TFLM int8 (reference) | 61,652,645 | 67,484,238 | 674,842 | 95,656 / 101,944 | 23,828 | yes |

### Per-layer cycles (mlperf_kws, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv0 / CONV_2D | 2,057,860 | 2,062,974 | 1,195,543 | 1,235,392 | 1,453,200 | 1,407,730 |
| dw1 / DEPTHWISE_CONV_2D | 701,124 | 451,291 | 701,176 | 451,289 | 720,546 | 727,835 |
| pw2 / CONV_2D | 2,946,842 | 2,938,761 | 1,244,510 | 1,235,756 | 1,318,660 | 1,297,432 |
| dw3 / DEPTHWISE_CONV_2D | 701,051 | 451,292 | 701,164 | 451,231 | 720,546 | 727,835 |
| pw4 / CONV_2D | 2,946,804 | 2,938,742 | 1,244,498 | 1,235,724 | 1,318,660 | 1,297,432 |
| dw5 / DEPTHWISE_CONV_2D | 701,054 | 451,238 | 701,164 | 451,287 | 720,546 | 727,835 |
| pw6 / CONV_2D | 2,946,813 | 2,938,748 | 1,244,497 | 1,235,673 | 1,318,663 | 1,297,435 |
| dw7 / DEPTHWISE_CONV_2D | 701,080 | 451,237 | 701,164 | 451,231 | 720,546 | 727,835 |
| pw8 / CONV_2D | 2,946,813 | 2,938,742 | 1,244,497 | 1,235,694 | 1,318,660 | 1,297,432 |
| avgpool9 / AVERAGE_POOL_2D | 21,185 | 21,432 | 21,180 | 21,433 | 75,830 | 67,608 |
| fc10 / FULLY_CONNECTED | 5,488 | 5,488 | 2,728 | 2,424 | 3,734 | 3,702 |
| softmax11 / SOFTMAX | 3,541 | 2,899 | 3,554 | 2,914 | 5,095 | 5,132 |

## MLPerf(R) Tiny reference image classification model (ResNet-8, 32x32x3 -> 10), 8 random int8 inputs

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 88,722,466 | 85,326,662 | 853,267 | 126,272 / 124,140 | 54,368 | yes |
| tinymind int8 (+SMLAD) | 27,697,964 | 26,373,751 | 263,738 | 103,564 / 112,884 | 54,440 | yes |
| TFLM int8 (CMSIS-NN) | 38,773,171 | 37,372,049 | 373,720 | 177,612 / 191,212 | 56,608 | yes |
| TFLM int8 (reference) | 159,042,196 | 191,555,451 | 1,915,555 | 144,516 / 155,412 | 55,396 | yes |

### Per-layer cycles (mlperf_ic, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv0 / CONV_2D | 3,179,116 | 3,186,799 | 2,164,564 | 1,920,078 | 2,846,244 | 2,837,416 |
| conv1 / CONV_2D | 16,702,995 | 16,612,851 | 5,089,274 | 5,021,092 | 7,276,144 | 6,984,937 |
| conv2 / CONV_2D | 16,702,946 | 16,612,802 | 5,089,188 | 5,021,011 | 7,270,334 | 6,979,128 |
| add3 / ADD | 704,725 | 442,426 | 704,722 | 442,426 | 1,381,687 | 1,407,630 |
| conv4 / CONV_2D | 8,401,198 | 8,070,345 | 2,594,584 | 2,546,234 | 3,316,766 | 3,171,053 |
| conv5 / CONV_2D | 16,441,616 | 15,844,833 | 4,447,999 | 4,414,716 | 6,010,446 | 5,718,998 |
| pw6 / CONV_2D | 856,587 | 924,186 | 518,576 | 490,899 | 679,028 | 677,296 |
| add7 / ADD | 352,469 | 213,057 | 352,470 | 213,057 | 688,519 | 702,260 |
| conv8 / CONV_2D | 8,291,343 | 7,945,039 | 2,272,012 | 2,239,415 | 2,893,540 | 2,747,445 |
| conv9 / CONV_2D | 16,252,638 | 17,968,081 | 4,249,534 | 4,222,241 | 5,506,920 | 5,213,962 |
| pw10 / CONV_2D | 964,188 | 829,221 | 443,126 | 450,111 | 530,106 | 533,502 |
| add11 / ADD | 184,531 | 102,456 | 184,530 | 102,456 | 343,791 | 350,923 |
| avgpool12 / AVERAGE_POOL_2D | 11,826 | 11,941 | 11,822 | 11,941 | 39,603 | 35,351 |
| fc13 / FULLY_CONNECTED | 4,583 | 4,580 | 2,334 | 2,088 | 3,275 | 3,235 |
| softmax14 / SOFTMAX | 3,574 | 2,866 | 3,556 | 2,877 | 5,015 | 5,048 |

## MLPerf(R) Tiny reference streaming wake word model (1D DS-CNN, 30 frames x 40 -> 3), 8 consecutive windows of random int8 frames; a step() row is one new 32 ms frame, the other rows one whole window

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 5,577,496 | 5,477,628 | 54,776 | 70,316 / 72,068 | 8,084 | yes |
| tinymind int8 (+SMLAD) | 2,175,387 | 2,121,407 | 21,214 | 73,380 / 78,040 | 8,192 | yes |
| tinymind int8 step(), one frame (scalar) | 367,283 | 348,939 | 3,489 | 67,148 / 68,080 | 4,224 | yes |
| tinymind int8 step(), one frame (+SMLAD) | 192,075 | 170,473 | 1,705 | 65,732 / 71,604 | 4,316 | yes |
| TFLM int8 (CMSIS-NN) | 2,937,874 | 2,917,973 | 29,180 | 156,336 / 165,920 | 17,984 | yes |
| TFLM int8 (reference) | 12,175,190 | 14,480,446 | 144,804 | 113,576 / 119,512 | 16,292 | yes |

### Per-layer cycles (mlperf_sww, each layer timed separately)

| Layer | tinymind +SMLAD O2 | tinymind +SMLAD O3 | tinymind step() +SMLAD O2 | tinymind step() +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| dwconv0 / DEPTHWISE_CONV_2D | 28,686 | 30,022 | 2,325 | 2,353 | 107,071 | 105,884 |
| conv1 / CONV_2D | 382,822 | 363,746 | 20,751 | 19,028 | 484,257 | 483,739 |
| dwconv2 / DEPTHWISE_CONV_2D | 112,095 | 105,912 | 13,123 | 8,736 | 333,420 | 329,232 |
| conv3 / CONV_2D | 883,624 | 878,165 | 47,631 | 46,185 | 976,413 | 966,683 |
| dwconv4 / DEPTHWISE_CONV_2D | 124,884 | 122,275 | 19,977 | 14,701 | 321,601 | 316,018 |
| conv5 / CONV_2D | 583,403 | 575,652 | 48,244 | 46,800 | 648,919 | 648,273 |
| dwconv6 / DEPTHWISE_CONV_2D | 28,699 | 19,462 | 28,690 | 19,699 | 29,621 | 29,080 |
| conv7 / CONV_2D | 11,774 | 11,315 | 11,643 | 11,241 | 32,948 | 35,361 |
| fc9 / FULLY_CONNECTED | 406 | 303 | 377 | 263 | 1,437 | 1,416 |
| softmax10 / SOFTMAX | 1,534 | 1,276 | 1,528 | 1,266 | 2,234 | 2,212 |


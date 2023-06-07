# Table of Contents

- [Table of Contents](#table-of-contents)
- [INT8 Rounding Modes](#int8-rounding-modes)
- [References](#references)

<br>

# INT8 Rounding Modes

<table class="tg">
<thead>
  <tr>
    <th class="tg-1wig" rowspan="2">Backend</th>
    <th class="tg-1wig" rowspan="2">Compute Kernel Quantization (FP32 to INT8)</th>
    <th class="tg-1wig" colspan="2">Weights Quantization (FP32 to INT8)</th>
  </tr>
  <tr>
    <th class="tg-1wig">Quantizaed Network (QAT)</th>
    <th class="tg-1wig">Dynamic Range API / Calibration</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">GPU</td>
    <td class="tg-0lax"><code>round-to-nearest-with-ties-to-even</code></td>
    <td class="tg-0lax"><code>round-to-nearest-with-ties-to-even</code></td>
    <td class="tg-0lax"><code>round-to-nearest-with-ties-to-positive-infinity</code></td>
  </tr>
  <tr>
    <td class="tg-0lax">DLA</td>
    <td class="tg-0lax"><code>round-to-nearest-with-ties-away-from-zero</code></td>
    <td class="tg-0lax">Not Applicable</td>
    <td class="tg-0lax"><code>round-to-nearest-with-ties-away-from-zero</code></td>
  </tr>
</tbody>
</table>


<br>

# References

- [NVIDIA TensorRT Documentation: INT8 Rounding Modes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-rounding-modes)
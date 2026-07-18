# Reference Benchmarks: cholupdate

**Date:** 2026-07-18

**Machine:**
- Architecture: arm64
- OS: macOS 26.5.2
- Python: 3.12.13

## Results

```
cholupdate micro-benchmark (mean per call)
     p   numpy (ms)  choldate (ms)  numpy/choldate
    10       0.0278         0.0006            49.7
    50       0.1346         0.0022            62.4
   100       0.2716         0.0072            37.9
   250       0.7090         0.0359            19.7
   500       1.5132         0.1415            10.7
  1000       3.4433         0.5547             6.2

swapping_css end-to-end (p=100, k=10, single init)
numpy backend:    0.005 s
choldate backend: 0.003 s
```

# Inspecting Input Data

The `inspect data` subtool can display information about input data generated
by a data loader.

For example, first we'll generate some input data by running inference:

```bash
polygraphy run identity.onnx --onnxrt --save-inputs inputs.json
```

Next, we can inspect them:

```bash
polygraphy inspect data inputs.json --show-values
```

This will display something like:

```
[I] ==== Data (1 iterations) ====

    x [dtype=float32, shape=(1, 1, 2, 2)]
        [[[[4.17021990e-01 7.20324516e-01]
           [1.14374816e-04 3.02332580e-01]]]]

    -- Statistics --
    x | Stats
        mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1)
```

# Inspecting A TensorRT Network

The `inspect model` subtool can automatically convert supported formats
into TensorRT networks, and then display them.

For example:

```bash
polygraphy inspect model identity.onnx \
    --mode=basic --display-as=trt
```

This will display something like:

```
[I] ==== TensorRT Network ====
    Name: Unnamed Network 0 | Explicit Batch Network

    ---- 1 Network Input(s) ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Network Output(s) ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Layer(s) ----
    Layer 0    | (Unnamed Layer* 0) [Identity] [Op: LayerType.IDENTITY]
        {x [dtype=float32, shape=(1, 1, 2, 2)]}
         -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
```

It is also possible to show detailed layer information, including layer attributes, using `--mode=full`.

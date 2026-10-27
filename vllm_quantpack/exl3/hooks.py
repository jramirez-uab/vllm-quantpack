"""Post-load hooks for model families that need to adjust EXL3 layers after their weights are in place.

Each hook is called once per layer at the end of process_weights_after_loading:

  LINEAR: hook(prefix, layer)        layer.exl3_shards / exl3_dense are final; may set layer.exl3_out_mul
  MOE:    hook(prefix, layer, ex)    ex is the layer's _Experts; may set ex.out_scale

A hook must return without touching the layer when the prefix or model is not one it handles.
"""
LINEAR: list = []
MOE: list = []

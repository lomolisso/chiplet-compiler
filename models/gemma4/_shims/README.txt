Import-only stand-ins for packages that google-deepmind/gemma imports but that
are not installed in .venv-graphs (kauldron pulls ~120 packages incl.
tensorflow and downgrades protobuf in the shared env). They provide ONLY the
names the gemma4 model files touch at import / annotation time:
  kauldron.ktyping / kauldron.typing : shape-annotated array types (Float['*B L'])
                                       and a no-op `typechecked` decorator
  kauldron.kd                        : kd.nn.Identity (flax identity module),
                                       kd.sharding.with_sharding_constraint (no-op)
  kauldron.kontext                   : Key / REQUIRED / Path (config plumbing)
  kauldron.utils.immutabledict       : freeze_dict_attrs (no-op; tokenizer class)
  dialog                             : Format enum (tokenizer class attribute)
  jaxtyping                          : _array_types.AbstractArray (isinstance probe)
None of them performs computation that reaches the traced graph. This directory
is appended to the END of sys.path, so real installs always take precedence.

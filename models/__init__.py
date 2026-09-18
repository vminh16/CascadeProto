"""CascadeProto models package.

Submodules are imported explicitly (e.g. `from models.eppm import EPPMStage`) so that
the pure-PyTorch modules load without `mamba_ssm` and `pointnet2_ops`; only
`models.encoder` and the modules built on it require them.
"""

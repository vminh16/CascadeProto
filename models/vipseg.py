"""
Legacy VIPSeg compatibility module.
Re-exports VIPSegBackbone and GatingNetwork from models.vipseg_backbone.
"""

from models.vipseg_backbone import VIPSegBackbone, GatingNetwork, extract_point_prototypes

__all__ = ['VIPSegBackbone', 'GatingNetwork', 'extract_point_prototypes']
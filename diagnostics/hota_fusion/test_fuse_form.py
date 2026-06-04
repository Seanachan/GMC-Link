"""Unit tests for the F1 nonlinear fusion term. Self-running (no pytest dep):
    python diagnostics/hota_fusion/test_fuse_form.py
"""
import sys
sys.path.insert(0, "/home/seanachan/GMC-Link")
from run_ikun_linear_additive import f1_extra


def test_f1_superset_zero():
    # beta=gamma=0 => F1 adds nothing => strict superset of F0
    assert f1_extra(2.0, 0.5, 0.0, 0.0) == 0.0
    assert f1_extra(-3.0, 0.8, 0.0, 0.0) == 0.0


def test_f1_interaction_term():
    # beta*(native*gmc) = 1.0*(2.0*0.5) = 1.0 ; gamma=0
    assert f1_extra(2.0, 0.5, 1.0, 0.0) == 1.0


def test_f1_curvature_term():
    # gamma*(gmc^2) = 2.0*(0.5*0.5) = 0.5 ; beta=0
    assert f1_extra(10.0, 0.5, 0.0, 2.0) == 0.5


def test_f1_both_terms():
    # beta*(native*gmc) + gamma*(gmc^2) = 1.0*(2.0*0.5) + 1.0*(0.25) = 1.25
    assert f1_extra(2.0, 0.5, 1.0, 1.0) == 1.25


if __name__ == "__main__":
    test_f1_superset_zero()
    test_f1_interaction_term()
    test_f1_curvature_term()
    test_f1_both_terms()
    print("OK: all f1_extra tests passed")

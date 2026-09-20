"""RED 2.4b — PR2 voto temporal 3f por track-id 2/3 + veto bee."""
import pytest


def test_stable_promotion_two_of_three():
    from jetson.ai.temporal import TemporalVote

    v = TemporalVote(p_min=0.995, p_bee_max=0.001, min_frames=3, min_agree=2)
    assert v.update("t1", 0.997, 0.0005) is None
    assert v.update("t1", 0.996, 0.0004) is None
    decision = v.update("t1", 0.500, 0.0002)
    assert decision is True, "2/3 frames Pv>=0.995 same track-id must promote"


def test_bee_veto_blocks_promotion():
    from jetson.ai.temporal import TemporalVote

    v = TemporalVote(p_min=0.995, p_bee_max=0.001, min_frames=3, min_agree=2)
    v.update("t2", 0.997, 0.0005)
    v.update("t2", 0.998, 0.0004)
    decision = v.update("t2", 0.990, 0.050)  # high-conf bee frame
    assert decision is False, "doubt->bee: high-conf bee frame must veto (no-fire)"


def test_tracks_are_independent():
    from jetson.ai.temporal import TemporalVote

    v = TemporalVote(p_min=0.995, p_bee_max=0.001, min_frames=3, min_agree=2)
    v.update("a", 0.997, 0.0001)
    v.update("a", 0.997, 0.0001)
    assert v.update("a", 0.997, 0.0001) is True
    # track b con solo 1 frame no puede promover
    assert v.update("b", 0.999, 0.0001) is None

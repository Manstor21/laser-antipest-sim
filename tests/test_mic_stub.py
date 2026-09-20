"""RED test 1.3 — mems_mic stub: reserved topic, zero audio, no DSP."""


def test_mic_stub_reserved_topic_and_no_dsp():
    from jetson.sensors.mems_mic.stub import (
        DSP_ENABLED,
        RESERVED_TOPIC,
        MicStub,
    )

    assert RESERVED_TOPIC == "mems_mic/audio_raw"
    assert DSP_ENABLED is False
    stub = MicStub()
    assert stub.connect() is True
    assert stub.is_connected is True
    frames = stub.read(num_frames=16)
    assert len(frames) == 0  # stub: zero audio processing
    assert stub.processed_samples == 0


def test_mic_stub_does_not_process_audio():
    from jetson.sensors.mems_mic.stub import MicStub

    stub = MicStub()
    stub.connect()
    # Any DSP entry point must refuse — stub only, no processing.
    try:
        stub.process([0] * 8)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "stub.process must raise NotImplementedError (no DSP)"

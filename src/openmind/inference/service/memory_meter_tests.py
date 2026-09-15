from openmind.inference.service.memory_meter import MemoryMeter


def test_the_meter_reads_what_this_process_holds() -> None:
    before = MemoryMeter().resident_bytes()
    ballast = bytearray(64 * 1024**2)

    after = MemoryMeter().resident_bytes()

    assert before > 1024**2
    assert after >= before + 32 * 1024**2 and len(ballast) == 64 * 1024**2

"""Device selection.

Windows lists one physical adapter several times -- once per driver stack --
under names that differ only because MME truncates at 31 characters. Treating
those as separate devices is what makes "which one is my phone adapter?"
unanswerable, so most of these tests are about collapsing them back together.
"""

import pytest

from call_transcriber import audio
from call_transcriber.audio import AudioError, InputDevice

# What a real Windows box reports for one USB telephone adapter plus onboard
# Realtek sound. Note the MME entry's name is cut short at 31 characters.
WINDOWS_LISTING = [
    InputDevice(1, "Microphone (USB PnP Audio Devic", 2, 44100, "MME"),
    InputDevice(4, "Primary Sound Capture Driver", 2, 44100, "Windows DirectSound"),
    InputDevice(5, "Microphone (USB PnP Audio Device)", 2, 44100, "Windows DirectSound"),
    InputDevice(9, "Microphone (USB PnP Audio Device)", 1, 48000, "Windows WASAPI"),
    InputDevice(10, "Stereo Mix (Realtek HD Audio Stereo input)", 2, 44100, "MME"),
    InputDevice(12, "Microphone (Realtek HD Audio Mic input)", 2, 44100, "MME"),
    InputDevice(13, "Line In (Realtek HD Audio Line input)", 2, 44100, "MME"),
    InputDevice(16, "Microphone (USB PnP Audio Device)", 2, 44100, "Windows WDM-KS"),
]


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(audio, "list_input_devices", lambda: list(WINDOWS_LISTING))


# -- identity --------------------------------------------------------------


def test_an_mme_truncated_name_matches_its_full_version():
    truncated = InputDevice(1, "Microphone (USB PnP Audio Devic", 2, 44100, "MME")
    full = InputDevice(9, "Microphone (USB PnP Audio Device)", 1, 48000, "Windows WASAPI")
    assert truncated.identity == full.identity


def test_genuinely_different_devices_do_not_share_an_identity():
    usb = InputDevice(9, "Microphone (USB PnP Audio Device)", 1, 48000, "Windows WASAPI")
    realtek = InputDevice(12, "Microphone (Realtek HD Audio Mic input)", 2, 44100, "MME")
    assert usb.identity != realtek.identity


def test_the_listing_shows_which_driver_stack_it_is():
    assert "Windows WASAPI" in str(WINDOWS_LISTING[3])
    assert "[9]" in str(WINDOWS_LISTING[3])


# -- picking a driver stack ------------------------------------------------


def test_wasapi_wins_when_available():
    """It reports the device's real format; the legacy stacks make one up."""
    entries = [d for d in WINDOWS_LISTING if "USB PnP" in d.name]
    assert audio.best_hostapi(entries).index == 9


def test_it_falls_back_down_the_preference_order():
    entries = [
        InputDevice(1, "Mic", 2, 44100, "MME"),
        InputDevice(5, "Mic", 2, 44100, "Windows DirectSound"),
    ]
    assert audio.best_hostapi(entries).index == 5


def test_an_unrecognised_stack_is_still_usable():
    entries = [InputDevice(3, "Mic", 2, 44100, "Some Future API")]
    assert audio.best_hostapi(entries).index == 3


# -- finding the adapter ---------------------------------------------------


def test_four_entries_for_one_adapter_resolve_to_one_device(windows):
    """The case that blocked setup: 'USB PnP' hits four rows, one adapter."""
    device = audio.find_device("USB PnP")
    assert device.index == 9
    assert device.hostapi == "Windows WASAPI"


def test_matching_is_case_insensitive(windows):
    assert audio.find_device("usb pnp").index == 9


def test_a_name_matching_two_real_devices_is_an_error(windows):
    with pytest.raises(AudioError) as exc:
        audio.find_device("Microphone")

    # The USB adapter and the Realtek mic. "Stereo Mix" and "Line In" are
    # also Realtek inputs but do not carry the word Microphone.
    assert "2 different devices" in str(exc.value)
    assert "device_index" in str(exc.value)


def test_no_match_lists_what_is_actually_available(windows):
    with pytest.raises(AudioError) as exc:
        audio.find_device("LRX")

    assert "no input device matching 'LRX'" in str(exc.value)
    assert "Realtek" in str(exc.value)


def test_an_explicit_index_overrides_name_matching(windows):
    assert audio.find_device("Realtek", index=5).index == 5


def test_an_index_that_no_longer_exists_explains_why(windows):
    with pytest.raises(AudioError) as exc:
        audio.find_device("USB PnP", index=99)

    assert "not an input device" in str(exc.value)
    assert "plugged or unplugged" in str(exc.value)


def test_a_negative_index_means_match_by_name(windows):
    assert audio.find_device("USB PnP", index=-1).index == 9


def test_an_empty_match_string_is_rejected(windows):
    with pytest.raises(AudioError, match="device_match is empty"):
        audio.find_device("   ")


def test_no_inputs_at_all_says_so(monkeypatch):
    monkeypatch.setattr(audio, "list_input_devices", lambda: [])
    with pytest.raises(AudioError, match="is the adapter plugged in"):
        audio.find_device("USB PnP")

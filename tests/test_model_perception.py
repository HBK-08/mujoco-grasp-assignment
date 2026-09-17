import base64

import numpy as np

from graspbench.perception import FoundationModelPerception


def test_grounding_json_parser_accepts_fenced_service_output() -> None:
    value = FoundationModelPerception._parse_json_object(
        '```json\n{"target_id":"blue_box","reason":"visible SAM3 candidate"}\n```'
    )
    assert value["target_id"] == "blue_box"


def test_sam3_packed_mask_protocol_round_trip() -> None:
    expected = np.zeros((2, 4, 5), dtype=bool)
    expected[0, 1:3, 2:4] = True
    expected[1, 0, 0] = True
    packed = np.packbits(expected.reshape(-1), bitorder="little")
    encoded = base64.b64encode(packed.tobytes()).decode("ascii")
    actual = FoundationModelPerception._decode_masks(encoded, 2, (4, 5))
    np.testing.assert_array_equal(actual, expected)

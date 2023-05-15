import pytest
import tiktoken

from gpt_json.types_streaming import StreamEventEnum

enc = tiktoken.get_encoding("cl100k_base")

def _tokenize(text):
    return [enc.decode([tok]) for tok in enc.encode(text)]

def _get_expected_stream_partial_objs(full_object):
    if isinstance(full_object, list):
        outer_partial = []
        yield outer_partial[:], StreamEventEnum.OBJECT_CREATED, None, None
        for value in full_object:
            for inner_partial, event, _, value_change in _get_expected_stream_partial_objs(value):
                if event == StreamEventEnum.OBJECT_CREATED:
                    outer_partial.append(inner_partial)
                    continue
                outer_partial[-1] = inner_partial
                yield (outer_partial[:], event, len(outer_partial) - 1, value_change)
    elif isinstance(full_object, dict):
        # get initial values for all keys first 
        value_iterators = {
            k: _get_expected_stream_partial_objs(v) for k, v in full_object.items()
        }
        outer_partial = {
            k : next(v)[0] for k, v in value_iterators.items()
        }
        yield outer_partial.copy(), StreamEventEnum.OBJECT_CREATED, None, None
        for key, value in full_object.items():
            for inner_partial, event, _, value_change in value_iterators[key]:
                # note: we've already consumed the first value (OBJECT_CREATED) for each key
                outer_partial[key] = inner_partial
                yield outer_partial.copy(), event, key, value_change
    elif isinstance(full_object, str):
        cumulative_str = ""
        yield cumulative_str, StreamEventEnum.OBJECT_CREATED, None, None

        tokens = _tokenize(full_object)
        for idx, token in enumerate(tokens):
            cumulative_str += token
            event = StreamEventEnum.KEY_COMPLETED if idx == len(tokens) - 1 else StreamEventEnum.KEY_UPDATED
            yield cumulative_str, event, None, token
    else: # number, bool
        yield None, StreamEventEnum.OBJECT_CREATED, None, None
        yield full_object, StreamEventEnum.KEY_COMPLETED, None, full_object

EXAMPLE_DICT = {'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': 'Good!' }

EXAMPLE_DICT_STREAM_DATA = [
    ({'student_model': '', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.OBJECT_CREATED, None, None),
    ({'student_model': 'The', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_UPDATED, 'student_model', 'The'),
    ({'student_model': 'The student', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_UPDATED, 'student_model', ' student'),
    ({'student_model': 'The student.', 'student_correct': None, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_COMPLETED, 'student_model', '.'),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': None, 'tutor_response': ''}, StreamEventEnum.KEY_COMPLETED, 'student_correct', False),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': ''}, StreamEventEnum.KEY_COMPLETED, 'correct_answer_number', 12156),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': 'Good'}, StreamEventEnum.KEY_UPDATED, 'tutor_response', 'Good'),
    ({'student_model': 'The student.', 'student_correct': False, 'correct_answer_number': 12156, 'tutor_response': 'Good!'}, StreamEventEnum.KEY_COMPLETED, 'tutor_response', '!')
]

EXAMPLE_STR_LIST = ["The student.", "Good!"]

EXAMPLE_STR_LIST_STREAM_DATA = [
    ([], StreamEventEnum.OBJECT_CREATED, None, None),
    (["The"], StreamEventEnum.KEY_UPDATED, 0, "The"),
    (["The student"], StreamEventEnum.KEY_UPDATED, 0, " student"),
    (["The student."], StreamEventEnum.KEY_COMPLETED, 0, "."),
    (["The student.", "Good"], StreamEventEnum.KEY_UPDATED, 1, "Good"),
    (["The student.", "Good!"], StreamEventEnum.KEY_COMPLETED, 1, "!")
]

EXAMPLE_NUMBER_LIST = [1356, 569834, 123]

EXAMPLE_NUMBER_LIST_STREAM_DATA = [
    ([], StreamEventEnum.OBJECT_CREATED, None, None),
    ([1356], StreamEventEnum.KEY_COMPLETED, 0, 1356),
    ([1356, 569834], StreamEventEnum.KEY_COMPLETED, 1, 569834),
    ([1356, 569834, 123], StreamEventEnum.KEY_COMPLETED, 2, 123)
]

EXAMPLE_MANY_LIST = ["The student.", 56789, False, 12.356]

EXAMPLE_MANY_LIST_STREAM_DATA = [
    ([], StreamEventEnum.OBJECT_CREATED, None, None),
    (["The"], StreamEventEnum.KEY_UPDATED, 0, "The"),
    (["The student"], StreamEventEnum.KEY_UPDATED, 0, " student"),
    (["The student."], StreamEventEnum.KEY_COMPLETED, 0, "."),
    (["The student.", 56789], StreamEventEnum.KEY_COMPLETED, 1, 56789),
    (["The student.", 56789, False], StreamEventEnum.KEY_COMPLETED, 2, False),
    (["The student.", 56789, False, 12.356], StreamEventEnum.KEY_COMPLETED, 3, 12.356)
]

@pytest.mark.parametrize(
    "full_object,expected_events",
    [
        (EXAMPLE_DICT, EXAMPLE_DICT_STREAM_DATA),
        (EXAMPLE_STR_LIST, EXAMPLE_STR_LIST_STREAM_DATA),
        (EXAMPLE_NUMBER_LIST, EXAMPLE_NUMBER_LIST_STREAM_DATA),
        (EXAMPLE_MANY_LIST, EXAMPLE_MANY_LIST_STREAM_DATA),
    ]
)
def test_get_expected_stream_objs(full_object, expected_events):
    assert list(_get_expected_stream_partial_objs(full_object)) == expected_events

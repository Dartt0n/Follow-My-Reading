import logging
from pathlib import Path

from typing import Any, Dict, List
from huey import RedisHuey

from core.plugins import (
    AUDIO_PLUGINS,
    IMAGE_PLUGINS,
    AudioProcessingResult,
    ImageProcessingResult,
    load_plugins,
)
from core.plugins.base import (
    AudioSegment,
    AudioTaskResult,
    TaskResult,
    TextDiff,
    ImageTaskResult,
    AudioProcessingFunction,
    ImageProcessingFunction,
)
from core.plugins.loader import PluginInfo
from core.processing.text import match_phrases
from core.processing.audio_split import split_audio
from pydub import silence, AudioSegment as PydubAudioSegments
from uuid import uuid4

scheduler = RedisHuey()


plugins = []


@scheduler.on_startup()
def load_plugins_into_memories() -> None:
    """
    Load plugins on startup. This function is introduced
    in order not to load plugins into the module on import.
    """
    global plugins
    plugins = load_plugins()


logger = logging.getLogger("huey")


def _plugin_class_method_call(class_name: str, function: str, filepath: str) -> Any:
    """
    `_plugin_class_method_call` is a function, which search each plugin for `class_name`
    object. If the object is not found, it raises KeyError. If found, the function
    gets the class and loads the `function` from it. According to `AudioProcessingPlugin`
    and `ImageProccesingPlugin` this function must be `@staticmethod`. Then,
    `_plugin_class_method_call` calls the loaded function with `filepath` argument and
    returns the result.
    """
    logger.info(f"Searching target plugin, which contains {class_name}")
    target = None
    # look through all loaded plugin
    for plugin in plugins:
        # if any plugin contain specified class, use this plugin as a targer
        if hasattr(plugin, class_name):
            logger.info(f"Target plugin found: {plugin.__name__}")
            target = plugin
            break
    else:
        # if no plugin is found, raise an error
        logger.info("Target plugin not found")
        raise KeyError(f"No plugin contain class {class_name}")

    logger.info(f"Getting class object ({class_name}) from target plugin")
    cls = getattr(target, class_name)  # load class from plugin module
    logger.info(f"Getting fuction ({function}) from class")
    func = getattr(cls, function)  # load function from class
    logger.info(f"Executing function {function} with {filepath=}")
    return func(filepath)  # call the function


@scheduler.task()
def dynamic_plugin_call(class_name: str, function: str, filepath: str) -> Any:
    """
    `dynamic_plugin_call` is a sheduled job, which accepts `class_name` (str),
    `function` (str), `filepath` (str) and returns the result of calling
    `_plugin_class_method_call` with these parameters.
    """
    return _plugin_class_method_call(class_name, function, filepath)


def _audio_process(audio_class: str, audio_path: str) -> AudioTaskResult:
    logger.info("Executing audio processing")
    audio_model_response: AudioProcessingResult = _plugin_class_method_call(
        audio_class, AudioProcessingFunction, audio_path
    )

    segments = []
    if len(audio_model_response.segments) != 0:
        audio_splits = [(s.start, s.end) for s in audio_model_response.segments]
        audio_files = split_audio(audio_path, audio_splits)

        for index, file_id in enumerate(audio_files):
            segments.append(
                AudioSegment(
                    start=audio_model_response.segments[index].start,
                    end=audio_model_response.segments[index].end,
                    text=audio_model_response.segments[index].text,
                    file=file_id,
                )
            )

        return AudioTaskResult(text=audio_model_response.text, segments=segments)

    # todo: use split silence on models, which can not split audio
    return AudioTaskResult(text=audio_model_response.text, segments=[])


@scheduler.task()
def audio_processing_call(audio_class: str, audio_path: str) -> AudioTaskResult:
    return _audio_process(audio_class, audio_path)


def _image_process(image_class: str, image_path: str) -> ImageTaskResult:
    logger.info("Executing image processing")
    image_model_response: ImageProcessingResult = _plugin_class_method_call(
        image_class, ImageProcessingFunction, image_path
    )
    return ImageTaskResult.parse_obj(image_model_response.dict())


@scheduler.task()
def image_processing_call(image_class: str, image_path: str) -> ImageTaskResult:
    return _image_process(image_class, image_path)


@scheduler.task()
def compare_image_audio(
    audio_class: str,
    audio_function: str,
    audio_path: str,
    image_class: str,
    image_function: str,
    image_path: str,
) -> TaskResult:
    """
    `compare_image_audio` is a scheduled job, which accepts these parameters:
    - `audio_class: str`
    - `audio_function: str`
    - `audio_path: str`
    - `image_class: str`
    - `image_function: str`
    - `image_path: str`

    Then `compare_image_audio` calls `_plugin_class_method_call` two times: for audio
    and for image correspondingly. When both of the calls are completed, it matches
    resulted texts and returns the difference.

    Note: with increased amount of workers, this job can call `dynamic_plugin_call`
    instead of `_plugin_class_method_call` and execute code simultaneously for
    audio and image processing.
    """
    audio_model_response: AudioTaskResult = _audio_process(audio_class, audio_path)
    image_model_response: ImageProcessingResult = _image_process(
        image_class, image_path
    )

    logger.info("Text matching")
    phrases = [x.text for x in audio_model_response.segments]
    text_diffs = match_phrases(phrases, image_model_response.text)

    data = []
    for index, diff in enumerate(text_diffs):
        for at_char, found, expected in diff:
            data.append(
                TextDiff(
                    audio_segment=audio_model_response.segments[index],
                    at_char=at_char,
                    found=found,
                    expected=expected,
                )
            )

    return TaskResult(
        audio=audio_model_response, image=image_model_response, errors=data
    )


@scheduler.task()
def _get_audio_plugins() -> Dict[str, PluginInfo]:
    """
    `get_audio_plugins` is a scheduled job, which returns info about
    loaded into the worker audio plugins.
    """
    return AUDIO_PLUGINS


@scheduler.task()
def _get_image_plugins() -> Dict[str, PluginInfo]:
    """
    `get_image_plugins` is scheduled job, which returns info about
    loaded into the worker image plugins.
    """
    return IMAGE_PLUGINS


def _search_pharse(words: List[str], phrase: str) -> List[int] | None:
    search_words = phrase.split(" ")

    search_index = 0
    expected_word_index = 0

    indexes = []

    while search_index < len(words):
        while (
            search_index < len(words)
            and words[search_index] != search_words[expected_word_index]
        ):
            search_index += 1

        stage_index = search_index

        while (
            search_index < len(words)
            and expected_word_index < len(search_words)
            and words[search_index] == search_words[expected_word_index]
        ):
            indexes.append(search_index)

            search_index += 1
            expected_word_index += 1

        else:
            search_index = stage_index
            expected_word_index = 0

        if expected_word_index == len(search_words):
            return indexes

        if search_index == len(words):
            return None

    return None


def _split_words(audio_file: str, audio_model: str) -> List[AudioSegment]:
    # load original audio into the memory.
    original_audio: PydubAudioSegments = PydubAudioSegments.from_file(audio_file)  # type: ignore
    logger.info(f"Loaded audio: {len(original_audio)} ms")
    logger.info("Splitting into segments")
    # split into non-silent segments, which are most likely words
    original_audio_segments: List[List[int]] = silence.detect_nonsilent(
        original_audio, min_silence_len=10, silence_thresh=-45
    )
    logger.info(
        f"Splitted into {len(original_audio_segments)} segments: {original_audio_segments}"
    )
    # generate id for every audio segment
    audio_segments_ids = [uuid4() for _ in range(len(original_audio_segments))]

    store_path = Path("./temp_data/audio")  # todo: replace with config

    # list of processed segments
    processed_segments = []

    # iterate over segments
    for index, (start, end) in enumerate(original_audio_segments):
        # save segment onto the disk
        audio_segment_file = store_path / str(audio_segments_ids[index])
        original_audio[start:end].export(out_f=audio_segment_file.as_posix())
        # perform audio process task
        result: AudioTaskResult = _audio_process(audio_model, str(audio_segment_file))
        # add all segments from result into processed_segments

        for segment in result.segments:
            # if number of words is more then 1, split again, else append to the result list
            if len(segment.text.split(" ")) > 1:
                processed_segments.extend(
                    _split_words(
                        (store_path / str(segment.file)).as_posix(), audio_model
                    )
                )
            else:
                processed_segments.append(segment)

    return processed_segments


def _extract_phrases_from_audio(
    audio_model: str, audio_file: str, text_phrases: List[str]
) -> List[AudioSegment | None]:
    logger.info("Loading original audio")
    original_audio: PydubAudioSegments = PydubAudioSegments.from_file(audio_file)  # type: ignore
    store_path = Path("./temp_data/audio")  # todo: replace with config
    logger.info("Splitting into the words")
    processed_segments = _split_words(audio_file, audio_model)
    logger.info(f"Extracted {len(processed_segments)} words")
    words_list = [s.text for s in processed_segments]
    logger.info(f"Extracted words: {words_list}")
    audio_phrases: List[AudioSegment | None] = []

    for phrase in text_phrases:
        # search phrase in a list of words
        join_audio_segments_indexes = _search_pharse(words_list, phrase)

        # if not found, add None
        if join_audio_segments_indexes is None or len(join_audio_segments_indexes) == 0:
            audio_phrases.append(None)
            continue

        # if found, join audio
        joined_segment = AudioSegment(
            start=processed_segments[join_audio_segments_indexes[0]].start,
            end=processed_segments[join_audio_segments_indexes[-1]].end,
            text=" ".join(map(lambda p: p.text, processed_segments)),
            file=uuid4(),
        )

        # generate file
        original_audio[
            int(joined_segment.start * 1000) : int(joined_segment.end * 1000)
        ].export((store_path / str(joined_segment.file)).as_posix())

        audio_phrases.append(joined_segment)

    return audio_phrases


@scheduler.task()
def extract_phrases_from_audio(
    audio_model: str, audio_file: str, phrases: List[str]
) -> List[AudioSegment | None]:
    return _extract_phrases_from_audio(audio_model, audio_file, phrases)

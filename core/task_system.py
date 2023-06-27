import logging

from huey import RedisHuey

from core.plugins import load_plugins
from core.processing.text import match

scheduler = RedisHuey()

plugins = load_plugins()
logger = logging.getLogger("huey")


def _plugin_class_method_call(class_name: str, function: str, filepath: str):
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
def dynamic_plugin_call(class_name: str, function: str, filepath: str):
    """
    `dynamic_plugin_call` is a sheduled job, which accepts `class_name` (str),
    `function` (str), `filepath` (str) and returns the result of calling
    `_plugin_class_method_call` with these parameters.
    """
    return _plugin_class_method_call(class_name, function, filepath)


@scheduler.task()
def compate_image_audio(
    audio_class: str,
    audio_function: str,
    audio_path: str,
    image_class: str,
    image_function: str,
    image_path: str,
):
    """
    `compate_image_audio` is a sheduled job, which accepts these parameters:
    - `audio_class: str`
    - `audio_function: str`
    - `audio_path: str`
    - `image_class: str`
    - `image_function: str`
    - `image_path: str`

    Then `compate_image_audio` calls `_plugin_class_method_call` two times: for audio
    and for image correspondingly. When both of the calls are completed, it matches
    resultred texts and returns the difference.

    Note: with increased amount of workers, this job can call `dynamic_plugin_call`
    instread of `_plugin_class_method_call` and execute code simultaneously for
    audio and image processing.
    """
    logger.info("Executing audio processing")
    audio_text = _plugin_class_method_call(audio_class, audio_function, audio_path)
    logger.info("Executing image processing")
    image_text = _plugin_class_method_call(image_class, image_function, image_path)

    logger.info("Text matching")
    return match(image_text, audio_text)
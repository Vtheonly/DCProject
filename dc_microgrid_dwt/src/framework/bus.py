import logging
from collections import defaultdict
from typing import Callable, Type, Dict, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self._subscribers: Dict[Type, List[Callable]] = defaultdict(list)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def subscribe(self, event_type: Type, callback: Callable):
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed {callback.__name__} to {event_type.__name__}")

    def publish(self, event):
        event_type = type(event)
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                # Execute fast!
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in handler {callback.__name__}: {e}", exc_info=True, extra={"props": {"event": str(event)}})

    def publish_async(self, event):
        """Fire and forget on a separate thread (for logging/slow tasks)"""
        self._executor.submit(self.publish, event)

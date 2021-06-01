
import asyncio
import os
import sys

import pytest


if os.name == 'nt' and sys.version_info >= (3, 7):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest.fixture
def event_loop():
    # Make sure we test against a selector event loop
    # since pyzmq doesn't like the proactor loop.
    # This fixture is picked up by pytest-asyncio
    if os.name == 'nt' and sys.version_info >= (3, 7):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.SelectorEventLoop()
    try:
        yield loop
    finally:
        loop.close()

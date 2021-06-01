"""Testing utils for jupyter_client tests

"""
import os
pjoin = os.path.join
import sys
from unittest.mock import patch
from tempfile import TemporaryDirectory
from typing import Dict

import pytest
from jupyter_client import AsyncKernelManager, KernelManager


skip_win32 = pytest.mark.skipif(sys.platform.startswith('win'), reason="Windows")


class test_env(object):
    """Set Jupyter path variables to a temporary directory
    
    Useful as a context manager or with explicit start/stop
    """
    def start(self):
        self.test_dir = td = TemporaryDirectory()
        self.env_patch = patch.dict(os.environ, {
            'JUPYTER_CONFIG_DIR': pjoin(td.name, 'jupyter'),
            'JUPYTER_DATA_DIR': pjoin(td.name, 'jupyter_data'),
            'JUPYTER_RUNTIME_DIR': pjoin(td.name, 'jupyter_runtime'),
            'IPYTHONDIR': pjoin(td.name, 'ipython'),
            'TEST_VARS': 'test_var_1',
        })
        self.env_patch.start()
    
    def stop(self):
        self.env_patch.stop()
        self.test_dir.cleanup()
    
    def __enter__(self):
        self.start()
        return self.test_dir.name
    
    def __exit__(self, *exc_info):
        self.stop()

def execute(code='', kc=None, **kwargs):
    """wrapper for doing common steps for validating an execution request"""
    from .test_message_spec import validate_message
    if kc is None:
        kc = KC
    msg_id = kc.execute(code=code, **kwargs)
    reply = kc.get_shell_msg(timeout=TIMEOUT)
    validate_message(reply, 'execute_reply', msg_id)
    busy = kc.get_iopub_msg(timeout=TIMEOUT)
    validate_message(busy, 'status', msg_id)
    assert busy['content']['execution_state'] == 'busy'

    if not kwargs.get('silent'):
        execute_input = kc.get_iopub_msg(timeout=TIMEOUT)
        validate_message(execute_input, 'execute_input', msg_id)
        assert execute_input['content']['code'] == code

    return msg_id, reply['content']


class RecordCallMixin:
    method_calls: Dict[str, int] = {}

    def record(self, method_name: str) -> None:
        if method_name not in self.method_calls:
            self.method_calls[method_name] = 0
        self.method_calls[method_name] += 1

    def call_count(self, method_name: str) -> int:
        assert method_name in self.method_calls
        return self.method_calls[method_name]

    def reset_counts(self) -> None:
        for record in self.method_calls.keys():
            self.method_calls[record] = 0


class SyncKernelManagerSubclass(RecordCallMixin, KernelManager):

    def start_kernel(self, **kw):
        self.record('start_kernel')
        return super().start_kernel(**kw)

    def shutdown_kernel(self, now=False, restart=False):
        self.record('shutdown_kernel')
        return super().shutdown_kernel(now=now, restart=restart)

    def restart_kernel(self, now=False, **kw):
        self.record('restart_kernel')
        return super().restart_kernel(now=now, **kw)

    def interrupt_kernel(self):
        self.record('interrupt_kernel')
        return super().interrupt_kernel()

    def request_shutdown(self, restart=False):
        self.record('request_shutdown')
        return super().request_shutdown(restart=restart)

    def finish_shutdown(self, waittime=None, pollinterval=0.1):
        self.record('finish_shutdown')
        return super().finish_shutdown(waittime=waittime, pollinterval=pollinterval)

    def _launch_kernel(self, kernel_cmd, **kw):
        self.record('_launch_kernel')
        return super()._launch_kernel(kernel_cmd, **kw)

    def _kill_kernel(self):
        self.record('_kill_kernel')
        return super()._kill_kernel()

    def cleanup_resources(self, restart=False):
        self.record('cleanup_resources')
        super().cleanup_resources(restart=restart)


class AsyncKernelManagerSubclass(RecordCallMixin, AsyncKernelManager):
    """Used to test subclass hierarchies to ensure methods are called when expected.

       This class is also used to test deprecation "routes" that are determined by superclass'
       detection of methods.

       This class represents a current subclass that overrides "interesting" methods of AsyncKernelManager.
    """
    which_cleanup = ""  # cleanup deprecation testing

    async def start_kernel(self, **kw):
        self.record('start_kernel')
        return await super().start_kernel(**kw)

    async def shutdown_kernel(self, now=False, restart=False):
        self.record('shutdown_kernel')
        return await super().shutdown_kernel(now=now, restart=restart)

    async def restart_kernel(self, now=False, **kw):
        self.record('restart_kernel')
        return await super().restart_kernel(now=now, **kw)

    async def interrupt_kernel(self):
        self.record('interrupt_kernel')
        return await super().interrupt_kernel()

    def request_shutdown(self, restart=False):
        self.record('request_shutdown')
        return super().request_shutdown(restart=restart)

    async def finish_shutdown(self, waittime=None, pollinterval=0.1):
        self.record('finish_shutdown')
        return await super().finish_shutdown(waittime=waittime, pollinterval=pollinterval)

    async def _launch_kernel(self, kernel_cmd, **kw):
        self.record('_launch_kernel')
        return await super()._launch_kernel(kernel_cmd, **kw)

    async def _kill_kernel(self):
        self.record('_kill_kernel')
        return await super()._kill_kernel()

    def cleanup(self, connection_file=True):
        self.record('cleanup')
        super().cleanup(connection_file=connection_file)
        self.which_cleanup = 'cleanup'

    def cleanup_resources(self, restart=False):
        self.record('cleanup_resources')
        super().cleanup_resources(restart=restart)
        self.which_cleanup = 'cleanup_resources'


class AsyncKernelManagerWithCleanup(AsyncKernelManager):
    """Used to test deprecation "routes" that are determined by superclass' detection of methods.

       This class represents the older subclass that overrides cleanup().  We should find that
       cleanup() is called on these instances via TestAsyncKernelManagerWithCleanup.
    """

    def cleanup(self, connection_file=True):
        super().cleanup(connection_file=connection_file)
        self.which_cleanup = 'cleanup'

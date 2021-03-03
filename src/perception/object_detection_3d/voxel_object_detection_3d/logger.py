
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Logger:

    LOG_WHEN_SILENT = 3
    LOG_WHEN_NORMAL = 2
    LOG_WHEN_VERBOSE = 1

    def __init__(self, silent, verbose, logfile_path=None):
        super().__init__()

        self.silent = silent
        self.verbose = verbose
        self.logfile = None

        if logfile_path is not None:
            self.logfile = open(logfile_path, "a")

    def log(self, level, *entries, **kwargs):

        if self.silent:
            if level >= Logger.LOG_WHEN_SILENT:
                print(*entries, **kwargs)

                if self.logfile is not None:
                    print(*entries, **kwargs, file=self.logfile)
        elif self.verbose:
            if level >= Logger.LOG_WHEN_VERBOSE:
                print(*entries, **kwargs)

                if self.logfile is not None:
                    print(*entries, **kwargs, file=self.logfile)
        else:
            if level >= Logger.LOG_WHEN_NORMAL:
                print(*entries, **kwargs)

                if self.logfile is not None:
                    print(*entries, **kwargs, file=self.logfile)

    def close(self):
        if self.logfile is not None:
            self.logfile.close()

# ----------------------------------------------------------------------------
# PyWavefront
# Copyright (c) 2013 Kurt Yoder
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of PyWavefront nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
import codecs
import gzip
import logging
from pathlib import Path

from pywavefront.exceptions import PywavefrontException

logger = logging.getLogger("pywavefront")


def auto_consume(func):
    """Decorator for auto consuming lines when leaving the function"""
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        args[0].consume_line()
    return inner


class Parser:
    """This defines a generalized parse dispatcher; all parse functions
    reside in subclasses."""
    auto_post_parse = True

    def __init__(self, file_name, strict=False, encoding="utf-8"):
        """
        Initializer for the base parser
        :param file_name: Name and path of the file to read
        :param strict: Enable or disable strict mode
        """
        self.file_name = Path(file_name).resolve()
        self.strict = strict
        self.encoding = encoding
        self.dir = Path(file_name).parent

        self.dispatcher = self._build_dispatch_map()
        self.lines = self.create_line_generator()

        self.line = None
        self.values = None

    def create_line_generator(self):
        """
        Creates a generator function yielding lines in the file
        Should only yield non-empty lines
        """

        if self.file_name.suffix == ".gz":
            # FIXME: Converting to str for now for py34 compatibility
            gz = gzip.open(str(self.file_name), mode='rt', encoding=self.encoding)

            for line in gz.readlines():
                yield line

            gz.close()
        else:
            # FIXME: Converting to str for now for py34 compatibility
            file = open(str(self.file_name), mode='r', encoding=self.encoding)

            for line in file:
                yield line

            file.close()

    def next_line(self):
        """Read the next line from the line generator and split it"""
        self.line = next(self.lines)  # Will raise StopIteration when there are no more lines
        self.values = self.line.split()

    def consume_line(self):
        """
        Tell the parser we are done with this line.
        This is simply by setting None values.
        """
        self.line = None
        self.values = None

    def parse(self):
        """
        Parse all the lines in the obj file
        Determines what type of line we are and dispatch appropriately.
        """
        try:
            # Continues until `next_line()` raises StopIteration
            # This can trigger here or in parse functions in the subclass
            while True:
                # Only advance the parser if the previous line was consumed.
                # Parse functions reading multiple lines can end up reading one line too far,
                # so they return without consuming the line and we pick it up here
                if not self.line:
                    self.next_line()

                if self.line[0] == '#' or len(self.values) < 2:
                    self.consume_line()
                    continue

                self.dispatcher.get(self.values[0], self.parse_fallback)()
        except StopIteration:
            pass

        if self.auto_post_parse:
            self.post_parse()

    def post_parse(self):
        """Override to trigger operations after parsing is complete"""
        pass

    @auto_consume
    def parse_fallback(self):
        """Fallback method when parser doesn't know the statement"""
        if self.strict:
            raise PywavefrontException("Unimplemented OBJ format statement '%s' on line '%s'"
                                       % (self.values[0], self.line.rstrip()))
        else:
            logger.warning("Unimplemented OBJ format statement '%s' on line '%s'"
                            % (self.values[0], self.line.rstrip()))

    def _build_dispatch_map(self):
        """
        Build a dispatch map: {func name": func} dict
        This is to easily dispatch to each parse method.

        Parse methods must start with `parse_` to be registered.
        The suffix should be the name of the obj statement
        such as `parse_v` for vertex statements.
        """
        return {"_".join(a.split("_")[1:]): getattr(self, a)
                for a in dir(self)
                if a.startswith("parse_")}

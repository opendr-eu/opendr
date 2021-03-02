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

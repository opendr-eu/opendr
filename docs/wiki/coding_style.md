This page defines the global coding style and Python and C coding styles. C++ coding style is defined in a separate page.
  * [Generic Coding Style](#generic-coding-style)
  * [Python Coding Style](#python-coding-style)
  * [C Coding Style](#c-coding-style)
  * [C++ Coding Style](cpp_coding_style.md)

## Generic Coding Style

#### #CS1 Prefer C to C++

When creating new code, prefer C to C++ programming language, except if a C++ dependency is required or if a complex object oriented design, including inheritance and other C++ features are strictly needed. C++ has a great complexity overhead over C that should be avoided as much as possible. Even with a C++ project, prefer a C public interface.

### Code quality

#### #CS2 No premature optimization

Don't complexify algorithms or data structure for optimization reasons. Optimization should take place at the very stage of development, only after the bottlenecks in a standard use case has been identified.

#### #CS3 Don't write unused code

Don't declare or implement unused variables or functions, under the pretext they could be used later. Implement them only when you need them.

#### #CS4 Don't pollute namespace

Use global static variables and static functions (C/C++) wherever needed to avoid polluting the global namespace.

#### #CS5 Choose names wisely for function and variable

The name of a function or variable should be sufficiently explicit to avoid a descriptive comment.

#### #CS6 No abbreviations

Don't use abbreviations. Exceptions are: `n` for "number of", `init` for "initialize", `max` for "maximum" and `min` for "minimum". C++ example:

```C++
computeAverage();  // NOT: compAvg();
```

#### #CS7 Acronyms should be written as other words (no capital letters)

C++ example:
```C++
exportHtmlSource(); // NOT: exportHTMLSource();
```

C example:
```C
open_dvd_player(); // NOT: open_DVD_player();
```

#### #CS8 No useless comments

Don't comment unless strictly needed.

#### #CS9 Don't comment out code

Source code must not be commented out without an explanation. If commented out, the explanation must include one of the following keywords (allcaps) `TODO:` or `FIXME:`, so that we can easily find out and fix this later.

The only exception to this rule is commenting out debug statements, for example in C:
```C
  // printf(stderr, "n_camera = %d\n", n_camera);
```

#### #CS10 Provide object description

An object definition (header file in C/C++ or source file in other languages for an object, interface, module, etc.) should include a short comment describing what is the purpose of this object.

#### #CS11 Use simple header comments

In headers of files, comments do **not** include:
- copyright (it is redundant with the general copyright of Webots)
- author (it is maybe true for the first commit, but it becomes very quickly obsolete once someone changes the file)
- modifications (it is difficult to maintain and it is redundant with the change log)
- file name (it is redundant with the file name)
- date (we don' care about it)

## Python Coding Style


#### #CS100 Use PEP8

When not specified otherwise by our coding style rules, use the [PEP 8](https://www.python.org/dev/peps/pep-0008) standard.
Note: using [Atom](https://atom.io/) with the [linter-flake8 linter](https://atom.io/packages/linter-flake8) package ensures that we respect our Python coding styles: `apm install linter-flake8`.

#### #CS101 Don't exceed 128 character for line of code

Line size is limited to 128 characters (instead of 80 in PEP8).

#### #CS102 Don't use PEP257 comments on all functions

We don't force PEP257 comments on all functions (especially, simple constructors, getters, setters, etc.) and therefore don't use the `flake8-docstrings` Atom package. However, we strongly encourage to respect this rule when it makes sense.

## C Coding style

#### #CS200 Default to C++ coding style for C

If not specified otherwise by our coding style, use the [C++ coding style](cpp_coding_style.md)

#### #CS201 Use underscore case for symbol names

Variables and functions names use the `underscore_case` notation (for type names use [#CS301](cpp_coding_style.md#cs301-pascal-case-for-type-names)).

#### #CS202 Use uppercase underscore case for preprocessor constants and macros

Preprocessor constants and macros names use the `UPPERCASE_UNDERSCORE_CASE` notation.

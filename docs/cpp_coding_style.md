## C++ Coding Style

Table of contents:
- [Tools](#tools)
- [Naming Conventions](#naming-conventions)
 - [Specific Naming Conventions](#specific-naming-conventions)
- [Files](#files)
 - [Source Files](#source-files)
 - [Include Files and Include Statements](#include-files-and-include-statements)
- [Statements](#statements)
 - [Types](#types)
 - [Inheritance](#inheritance)
 - [Variables](#variables)
 - [Loops](#loops)
 - [Conditionals](#conditionals)
 - [Functions](#functions)
 - [Enums](#enums)
 - [Miscellaneous](#miscellaneous)
 - [White Space](#white-space)
 - [Comments](#comments)

***

### Tools

#### #CS300 OpenDR clang-format style compliance
In order to facilitate the respect of our coding style, developer should respect the style defined in our clang format configuration file: https://github.com/opendr-eu/opendr_internal/blob/master/.clang-format
`clang-format` may be used with the Atom text editor by installing the `atom-beautify` package (`apm install atom-beautify`) and setting `clang-format` as the default Beautifier for C and C++ languages.
Note that no warnings are printed and the beautify has to be executed manually or you should enable the apply on save options.

On Ubuntu, `clang-format` 6.0 can be installed from APT and then you have to install `clang-format` Atom package:

```
sudo apt remove clang-format clang-format-3.8  # if required
sudo apt install clang-format-6.0
sudo ln -s /usr/bin/clang-format-6.0 /usr/bin/clang-format
```

It may be needed to patch atom beautify with this: https://github.com/Glavin001/atom-beautify/issues/2290

On macOS, simply type:

```
brew install clang-format
apm install clang-format
```

... and to set `clang-format` into the `Atom / Preferences / Packages / Clang Format / Settings / Executable`.

`cppcheck` is also very useful to check for both the style and possible programming errors.
The latest version is available from http://cppcheck.sourceforge.net.

On Ubuntu you have to install `cppcheck` from sources to get the latest version.
Extract the package and type:
```
make MATCHCOMPILER=yes FILESDIR=/usr/share/cppcheck HAVE_RULES=yes
sudo make install FILESDIR=/usr/share/cppcheck/
```

On macOS, simply type:

```
brew install cppcheck
apm install linter-cppcheck
```

Unfortunately, the `linter-cppcheck` atom package seems to be currently broken and unsupported.
`cppcheck` is used in the CI sources tests with the following options:
  - Enable Information
  - Enable Missing Include
  - Enable Performance
  - Enable Portability
  - Enable Style
  - Enable Warning
  - Inline Suppressions

### Naming Conventions

#### #CS301 Pascal case for type names
Names representing types must be in `PascalCase` notation: mixed case starting with upper case.
```
Line, SavingsAccount
```

#### #CS302 Camel case for variable names
C++ Variable names must be in `camelCase`: mixed case starting with lower case.
```
int line;
SavingsAccount savingsAccount;
```

#### #CS303 Non-static private class variables should have the "m" prefix.
```
private:
  int mLineSize;
  SavingsAccount mSavingsAccount;
```

#### #CS304: Static private class variables should have the "c" prefix.
```
private:
  static int cLineSize;
  static SavingsAccount cSavingsAccount;
```

#### #CS305: Static variables (non-class members) should have the "g" prefix.
```
static int gLine;
```

#### #CS306 Global constant names (including enum values) must be all uppercase using underscore to separate words.
```
static const int MAX_ITERATIONS;
enum { RED, GREEN, BLUE };
```

#### #CS307 Names representing methods or functions must be written in mixed case starting with lower case.
```
QString name();
double computeTotalWidth();
```

#### #CS308 Names representing template types should be a single uppercase letter.
```
template<class T> ...
template<class C, class D> ...
```

#### #CS309 Generic variables should have the same name as their type.
```
void setTopic(Topic *topic)
// NOT: void setTopic(Topic *value)
// NOT: void setTopic(Topic *aTopic)
// NOT: void setTopic(Topic *t)

void connect(Database *database)
// NOT: void connect(Database *db)
// NOT: void connect (Database *oracleDB)
```

#### #CS310 Variables with a large scope should have long names, variables with a small scope can have shorter names.

#### #CS311 The name of the object is implicit, and should be avoided in a method name.
```
line.length(); // NOT: line.lineLength();
```

### Specific Naming Conventions

#### #CS312 The term set must be used where an attribute is changed directly.
```
employee.setName(name);
matrix.setElement(2, 4, value);
```

#### #CS313 Methods that return a value should be named after the value they return (without get prefix).
```
employee.name();  // NOT employee.getName();
matrix.element(2, 4);  // NOT matrix.getElement(2, 4);
```

#### #CS314 The `compute` prefix should be used in methods where something (complex) is computed.
```
valueSet->computeAverage();
matrix->computeInverse();
```

#### #CS315 The `find` prefix should be used in methods where something is looked up.
```
vertex.findNearestVertex();
matrix.findMinElement();
```

#### #CS316 The `init` prefix should be used where an object or a concept is established and match with a corresponding `cleanup` prefix.
```
printer.initFontSet();
:
printer.cleanupFontSet();

```

#### #CS317 Variables representing GUI components should be suffixed by the component type name.
```
mainWindow, propertiesDialog, widthScale, loginText, leftScrollbar, mainForm, fileMenu, minLabel, exitButton, yesToggle etc.
```

#### #CS318 Plural form should be used on names representing a collection of objects.
```
vector<Point> points;
int values[];
```

#### #CS319 The `n` prefix should be used for variables representing a number of objects.
```
nPoints, nLines
```

#### #CS320 Iterator variables should be called i, j, k etc.
```
for (int i = 0; i < nTables); ++i) {
  :
}

vector<MyClass>::iterator i;
for (i = list.begin(); i != list.end(); ++i) {
  Element element = *i;
  :
}
```

#### #CS321 The `is` prefix should be used for boolean variables and methods.
```
bool isSet() const { return mIsSet; }
bool isVisible() const;
bool isFinished() const;

private:
bool mIsFound;
bool mIsOpen;

// There are a few alternatives to the `is` prefix that fit better in some situations. These are the `has`, `can` and `should` prefixes:
bool hasLicense();
bool canEvaluate();
bool shouldSort();
```

#### #CS322 Complement names must be used for complement operations.
```
init/cleanup, add/remove, create/destroy, start/stop, insert/delete, increment/decrement, old/new, begin/end, first/last, up/down, min/max, next/previous, old/new, open/close, show/hide, suspend/resume, etc.
```

#### #CS323 Negated boolean names must be avoided.
```
bool hasErrors;   // NOT: hasNoError
bool isFound;     // NOT: isNotFound
bool isRunning()  // NOT: isNotRunning()
```

### Files

#### #CS324 Use pascal case for source files
C++ header files should have the `.hpp` extension. C++ source files should have the `.cpp` extension.
```
MyClass.cpp, MyClass.hpp
```

#### #CS325 A class should be declared in a header file and defined in a source file where the name of the files match the name of the class.
```
MyClass.hpp:

class MyClass {
 :
```

#### #CS326 Inline simple methods

Inlining should be used for methods that can be written in one line or so (typically getters and setters), if no additional include is required.
```
class MyClass {
  void setValue(Value value)  { mValue = value; }
  Value value() const         { return mValue; }
```

#### #CS327 Inline performance critical methods

For optimization reasons, longer methods can also be inlined, in this case the body of the method must be placed in the header file, just below the class declaration. Remember to avoid early optimization.

### Include Files and Include Statements

#### #CS328 Header files must contain an include guard
```
#ifndef CLASS_NAME_HPP
#define CLASS_NAME_HPP
 :
#endif
```

#### #CS329 Sort `#include` statements in header blocks

`#include` statements in a header or a source file should respect the following order, seperated by an empty line:
- the most meaningful header comes first, followed by an empty line; usually we include "MyClass.hpp" at the top of MyClass.cpp text
- toolkit headers come second in alphabetical order
- standard headers come last in alphabetical order
```
%From  MyClass.cpp

#include "MyClass.hpp"

#include "OtherClass1.hpp"
#include "OtherClass2.hpp"

#include <limits>
#include <cassert>
```

#### #CS330 Don't use an #include when a forward declaration would suffice
```
#include "MyOtherClass.hpp"  // NO

class MyOtherClass;  // YES
```

### Statements

#### #CS331 Types that are local to one file only can be declared inside that file.

#### #CS332 Use a struct only for passive objects that carry data; everything else is a class.

#### #CS333 The parts of a class must be sorted public, protected and private. All sections must be identified explicitly.
```
class MyClass {
public:
  ...
protected:
  ...
private:
  ...
}
```

### Inheritance

#### #CS334 Do not change the public/protected/private status of a method in derived classes.

#### #CS335 Use virtual in front of a derived function declaration if the base class function was declared virtual.

#### #CS336 Always declare destructors as virtual except in classes that are not meant to be derived.

#### #CS337 Declare a function virtual only if it is going to be used polymorphically.

#### #CS338 Use public inheritance exclusively.
```
class DerivedClass : public BaseClass {
```

### Variables

#### #CS339 Variables should be initialized where they are declared.
```
QString s = "abc";

int x, y, z;
computeCenter(&x, &y, &z);

if (x > z) {
  int w = x * z;
  :
}
```

#### #CS340 Member variables should be initialized in the constructor initialization list provided it doesn't yield code duplication in other constructors.
```
MyClass::MyClass() : mMySize(10), mMyObject1("obj1"), mMyObject2("obj2") {
  % non trivial initialization;
}
```

#### #CS341 Place a function's variables in the narrowest scope possible (block, function, class), and initialize variables in the declaration.
```
int i;
i = f();      // BAD - initialization separate from declaration.

int j = g();  // GOOD - declaration has initialization.
```

#### #CS342 Do not use global variables, use singletons or static methods instead.

#### #CS343 Use the copy constructor for an object definition
```
MyVector2 u(0.0, 1.0, 0.0);
MyVector2 v(u); // YES
MyVector2 w = u; // the very same thing as above, but NO
```

#### #CS344 Class variables should never be declared public. Use getters and setters instead.
```
// BAD!
class MyClass {
public:
  int numEmployee;
}

// GOOD
class MyClass {
public:
  void setNumberOfEmployees();
  int numberOfEmployees() const;

private:
  int mNumberOfEmployee;
}
```

#### #CS345 C++ pointers and references should have their reference symbol next to the name rather than to the type.
```
float *x, *y, *z;  // NOT: float* x, y, z;
int &y;            // NOT: int& y;
```

#### #CS346 Implicit test for 0 should not be used other than for boolean and pointer variables.
```
if (isFinished()) {
  :
}
```

#### #CS347 Do not test boolean expressions against true or false
```
// YES
if (isFinished()) ...
if (!isOpen()) ...

// NO
if (isFinished() == true) ...
if (isOpen() == false) ...
```

### Loops

#### #CS348 Loop variables should be initialized immediately before the loop.
```
bool isDone = false;
while (!isDone) {
  :
}
```

#### #CS349 The use of break and continue in loops should only be used if they give higher readability than their structured counterparts.

#### #CS350 The `while (true)` form should be used for infinite loops.
```
while (true) {
  :
}
```

### Conditionals

#### #CS351 The if-else class of statements should have the following form:
```
if (condition) {
  statements;
}

if (condition) {
  statements;
} else {
  statements;
}

if (condition) {
  statements;
} else if (condition) {
  statements;
} else {
  statements;
}
```

#### #CS352 A for statement should have the following form:
```
for (initialization; condition; update) {
  statements;
}
```

#### #CS353 A while statement should have the following form:
```
while (condition) {
  statements;
}
```

#### #CS354 Single `if`, `while` or `for` statement must be written without brackets. The statement part should be put on a separate line. In case of nested block keep brackets.
```
// BAD:
if (condition)
  for (initialization; condition; update)
    statement;

if (condition) statement;

// GOOD:
if (condition) {
  for (initialization; condition; update)
    statement;
}

if (condition)
  statement;

while (condition)
  statement;

for (initialization; condition; update)
  statement;
```

#### #CS355 Assignment in conditionals must be avoided.
```
// GOOD
File *fileHandle = open(fileName, "w");
if (!fileHandle) {
  :
}

// BAD
if (!(fileHandle = open(fileName, "w"))) {
  :
}
```

### Functions

#### #CS356 Write small and focused functions: the body of a function should not exceed one page (no scrolling).

#### #CS357 High-level and low-level code should not be mixed in a same function.

#### #CS358 When defining a function, parameter order is: inputs, then outputs. If a function returns only one value use the return type.
```
void computePoints(const Line &line, Point &p1, Point &p2);
Point computePoint(const Line &line1, const Line &line2);
```

#### #CS359 Prefer pre-incrementation over post-incrementation whenever possible.
```
vector<MyClass>::iterator i;
for (i = list.begin(); i != list.end(); ++i) { % NOT i++
  Element element = *i;
  :
}
```

### Enums

#### #CS360 Prefer implicit initialization.

```c++
enum {
  A, // implicitly 0
  B, // implicitly 1
  C = 12,
  D // implicitly 12
}
```

### Miscellaneous

#### #CS361 Prefer `int` data type for integer numbers, including unsigned integer numbers.

#### #CS362 Prefer `double` data type for floating point numbers.

#### #CS363 Floating point constants should be written with decimal point and at least one decimal.
```
double total = 0.0; // NOT: double total = 0;
double speed = 3.0e8; // NOT: double speed = 3e8;

double sum = (a + b) * 10.0;
```

#### #CS364 Use 0 for integers, 0.0 for reals, NULL for pointers, and '\0' for chars.
```
if (p == NULL && fabs(a) == 0.0 && size == 0)
  char c = '\0';
```

#### #CS365 Use C++ static_cast<>() instead of C style casts.
```
A *a = (A*)b;  // NO
A *a = static_cast<A*>(b);  // YES
```

#### #CS366 Use C++ dynamic_cast<>() to test the run-time type of an object.
```
A *a = dynamic_cast<A*>(b);
if (a)
  doSomethingWithA(a);
```

#### #CS367 Do never use the friend keyword (but for operator overloading).

#### #CS368 Avoid defining macros at all costs.

#### #CS369 Avoid operator overloading, except in these two cases:
  - To define mathematical classes, and only if the operators supports a known math syntax
  - To inter-operate with streams

#### #CS370 Use const whenever it makes sense to do so, e.g.,:
  - Declare const any method that does not change a variable of its class.
  - Declare const any pointer or reference parameter that will not be altered by a function.
  - Declare const any variable which value will not change.
```
double computeTotal(const double v[], int n) const;
const double size = 4.5;
```

### White Space

#### #CS371 Special characters like Tab and page break must be avoided.

#### #CS372 Basic indentation should be 2 spaces.

#### #CS373 Conventional operators should be surrounded by a space character. Commas and semi-colons (in for loops) should be followed by a white space.
```
a = (b + c) * d; // NOT: a=(b+c)*d
doSomething(a, b, c, d); // NOT: doSomething(a,b,c,d);
for (i = 0; i < 10; ++i) { // NOT: for(i=0;i<10;i++){ ...
```

#### #CS374 Logical units within a block should be separated by one blank line.
```
Matrix4x4 matrix = new Matrix4x4();

double cosAngle = Math.cos(angle);
double sinAngle = Math.sin(angle);

matrix.setElement(1, 1, cosAngle);
matrix.setElement(1, 2, sinAngle);
matrix.setElement(2, 1, -sinAngle);
matrix.setElement(2, 2, cosAngle);

multiply(matrix);
```

#### #CS375 Use alignment wherever it enhances readability.

### Comments

#### #CS376 Use // for all comments, including multi-line comments.

#### #CS377 Use /**/ only for temporary debugging purposes.

#### #CS378 Follow the standard indentation for a class declaration
```
class MyClass: public BaseClass {

public:
  MyClass();
  virtual ~MyClass();
  void setData(Data *data);

private:
  Data *mData;
}
```

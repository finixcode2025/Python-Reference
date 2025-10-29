# =================================================================
# Zahra's Complete Python Reference
# Author: Zahra
# Purpose: Full reference of Python concepts.
# ==================================================================

# ==========================================================================================
# VARIABLES & DATA TYPES
# ==========================================================================================
name = "Zahra"              # String
age = 16                    # Integer
height_cm = 160.5           # Float
is_student = True           # Boolean
print("String:", name)
print("Integer:", age)
print("Float:", height_cm)
print("Boolean:", is_student)

# ==========================================================================================  
# OPERATORS 
# ==========================================================================================

# 1.------------ Arithmetic Operators ------------ 
num_a, num_b = 10, 3

# Addition
assert num_a + num_b == 13

# Subtraction
assert num_a - num_b == 7

# Multiplication
assert num_a * num_b == 30

# Division
assert num_a / num_b == 10 / 3  # 3.333...

# Floor Division
assert num_a // num_b == 3

# Modulus
assert num_a % num_b == 1

# Exponentiation
assert num_a ** num_b == 1000


# 2.----------- Comparison/Relational Operators ----------
x = 5
y = 5

# Equal
assert x == y

# Not equal
assert (x != y) == False

# Less than
assert (x < y) == False

# Less than or equal
assert x <= y

# 3.----------- Logical Operators (and, or, not) -----------
# And
assert (True and False) == False

# Or
assert (True or False) == True

# Not
assert (not True) == False


# 4.------------ Assignment Operators ------------
value = 10                    # assignment
assert value == 10

value += 5                    # add and assign
assert value == 15

value -= 5                    # subtract and assign
assert value == 10

value *= 2                    # multiply and assign
assert value == 20

value /= 3                    # divide and assign
assert value == 20 / 3  # 6.666...

value //= 5                   # floor divide and assign
assert value == (20 / 3) // 5  # 1.0

value %= 3                    # modulus and assign
assert value == ((20 / 3) // 5) % 3  # 1.0

value **= 4                   # exponentiation and assign
assert value == (((20 / 3) // 5) % 3) ** 4  # 1.0


# 5.------------ Membership operators -----------
num_list = [1, 2, 3]
assert 2 in num_list
assert 5 not in num_list

word = "python"
assert "py" in word
assert "on" in word

data_map = {"a":1, "b":2}
assert "a" in data_map             # checks key
assert 1 in data_map.values()      # correct way to check value



# =========================================================================================  
# CONDITIONALS 
# =========================================================================================

# 1.---------- Basic condition ----------
print("\n--- Conditionals ---\n")

price_a = 20
price_b = 15

if price_a > price_b:
    print("\nPrice A is more expensive!")  
elif price_a == price_b:
    print("Price A equals Price B.")
else:
    print("Price B is more expensive!")
# output: Price A is more expensive!

# 2.-------------- Chained comparisons -------------
temp_now = 5
if 1 < temp_now <= 10:
     print("\nTemperature is between 1°C and 10°C")

# 3.--------------- with Logical ---------------
temp = 32
raining = False

if temp > 30 and not raining:
    print("\nExcessive hot and heat waves!")
else:
    print("Normal weather conditions.")
# output: Excessive hot and heat waves!

# 4.--------------- Membership (List/Tuple/Dictionary) ----------------
message = "Hello!"
if "He" in message:
    print("\n'He' exists in the message string.")          # 'He' exists in the message string.

char_map = {"A" : 1, "B" : 2}
if "A" in char_map:
    print(f"Key 'A' exists with value {char_map['A']}")    # Key 'A' exists with value 1

# 5.--------------------- Use identity comparison (is) for singleton -------------------
data_item = None
if data_item is None:
    print("\nIdentity Check: data_item is None.")

# Ternary/Conditional expression:
current_number  = 3
parity_label = "even" if current_number % 2 == 0 else "odd"
print("Conditional Expressions result:", parity_label)


# ==========================================================================================
# ERROR HANDLING
# ==========================================================================================

# 1. Specific Error
try:
    10 / 0
except ZeroDivisionError:
    print("Dividing by zero is impossible!")  # Output: Dividing by zero is impossible!

# 2. Any Error (generic)
try:
    print(10 / 0)
except Exception as error_message:
    print("Raise Error:", error_message)     # Output: Raise Error: division by zero

# 3. Finally and Else
try:
    5 / 1
except ZeroDivisionError:
    print("ZeroDivisionError!")
else:
    print("No Error")                       # Output: No Error
finally:
    print("Done")                           # Output: Done

# 4. Several Except
try: 
    user_input = int("Salam")
except ValueError:
    print("This is not a number!")          # Output: This is not a number!
except ZeroDivisionError:
    print("Dividing by Zero!")

# 5. Several Exception in one Except
try:
    user_value = int("Hello")
except (ValueError, TypeError) as exception_detail:
    print("Error:", exception_detail)       # Output: Error: invalid literal for int() with base 10: 'Hello'

# 6. Custom Error (Raise)
class AgeError(Exception):
    pass

def check_age(user_age):
    if user_age < 18:
        raise AgeError("Your age is not allowed!")
    return "Welcome!"

# Example usage:
try:
    print(check_age(20))                    # Output: Welcome!
except AgeError as e:
    print("Error:", e)

# 7. ExceptionGroup
def raise_multiple_errors():
    errors_list = [
        ValueError("Bad value!"),
        TypeError("Wrong type!"),
        KeyError("Missing key")
    ]
    raise ExceptionGroup("Multiple errors happened", errors_list)

try:
    raise_multiple_errors()
except ExceptionGroup as group_exception:
    print(f"Caught ExceptionGroup: {group_exception}")  
    # Output: Caught ExceptionGroup: Multiple errors happened (3 sub-exceptions)

# ====================================================================================================
# LOOPS
# ====================================================================================================

# ---------------- FOR LOOP ----------------

# 1. Simple Loop
for index in range(3):
    print("Looping with For:", index)  
    # Output: 0, 1, 2

# 2. Range(Start, Stop, Step)
for countdown in range(5, 0, -1):
    print("Contrary Counting Using For Loop:", countdown)  
    # Output: 5, 4, 3, 2, 1

# 3. Combining For Loop with Conditions
print("Combining 'For' Loop with conditions:")
for number in range(10):
    label = "even" if number % 2 == 0 else "odd"
    print(number, label)  
    # Example Output: 0 even, 1 odd, ...

# 4. Nested Loops (For)
print("Nested Loops:")
for row_index in range(1, 4):
    for col_index in range(1, 4):
        print(f"Row = {row_index}, Column = {col_index}")  
        # Output: Row = 1, Column = 1 ... Row = 3, Column = 3

# 5. Family Names Combination
print("My Family:")
family_names = ["Mohammad Jawad", "Zubaida", "Hadisa", "Rona", "Zahra", "Shukria"]
last_names = ["Nazari", "Sakhizadeh", "NazarZadeh"]

for first_name in family_names:
    for last_name in last_names:
        print(f"{first_name} {last_name}")  
        # Example Output: Mohammad Jawad Nazari, Mohammad Jawad Sakhizadeh, ...

# 6. For Loop with Dictionary
sample_dict = {"x": 10, "y": 20}

print("Dictionary with For Loop:")
for key in sample_dict:
    print(key, sample_dict[key])  
    # Output: x 10, y 20

print("For Loop with items():")
for key, value in sample_dict.items():
    print(key, value)  
    # Output: x 10, y 20

# 7. For Loop with String
print("For Loop with String:")
for char in "ZAHRA":
    print(char)  
    # Output: Z A H R A

# ---------------- WHILE LOOP ----------------

# 1. Simple While Loop
counter = 0
while counter < 3:
    print("While loop:", counter)  
    # Output: 0, 1, 2
    counter += 1

# 2. Nested While Loops (Multiplication Table)
print("Nested While Loops (Multiplication Table):")
multiplicand = 1
while multiplicand <= 3:
    multiplier = 1
    while multiplier <= 3:
        print(multiplicand, "*", multiplier, "=", multiplicand * multiplier)  
        # Output: 1 * 1 = 1 ... 3 * 3 = 9
        multiplier += 1
    multiplicand += 1



# =======================================================================================================
#  FUNCTIONS 
# =======================================================================================================
# 1. ------------ Function Basics: --------------

def say_hi() -> None:
    """Prints a simple greeting."""
    print("Hi!")

def greet(name: str) -> str:
    """Returns a personalized greeting."""
    return f"Hello {name}!"

def add_number(a: int, b: int) -> int:
    """Returns the sum of two numbers."""
    return a + b

def print_separator() -> None:
    """Utility function to print a separator."""
    print("Utility Function:\n---------")



# 2. ------------- Function Parameters:-------------

def additional_sum(a: int, b: int) -> int:
    """Positional parameters."""
    return a + b

def introduce(name: str, age: int) -> None:
    """Keyword arguments example."""
    print(f"My name is {name}, I'm {age} years old.")

def greet_user_default(name: str = "Friend") -> None:
    """Default argument."""
    print(f"Hello, {name}!")

def sum_arbitrary(*numbers: int) -> int:
    """Arbitrary positional arguments (*args)."""
    return sum(numbers)

def display_info(**info) -> None:
    """Arbitrary keyword arguments (**kwargs)."""
    print(info)

def flexible_example(a, b, *args, **kwargs) -> None:
    """Mix of positional, *args, and **kwargs."""
    print("a:", a)
    print("b:", b)
    print("args:", args)
    print("kwargs:", kwargs)


# 3. ---------------- Variable Scope: -----------------

# Local scope:
def local_example():
    x = 10
    print(x)

# Enclosing scope:
def enclosing_example():
    outer_var = "I am outer"

    def inner_func():
        print(outer_var)
    inner_func()

# Global scope:
global_var = 1000

def print_global():
    print(global_var)

# Built-in scope example:
print(len("hello"))


# 4.----------- Nested Functions ------------

def nested_example():
    """Outer and inner functions."""

    def inner():
        print("outer function")
    inner()


def modify_outer_var():
    x = 10

    def inner():
        nonlocal x
        x += 5
        print("Inner:", x)
    inner()
    print("outer after inner")


# 6. --------------- Decorators --------------

#  Basic decorator:

def decorator_basic(func):                     
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper
 
def greet_simple():         
    print("Hello")

decorated = decorator_basic(greet_simple)      
decorated()


# Decorator with Arguments:
def decorator_args(func):
    def wrapper(*args, **kwargs):
        print("Before running function")
        result = func(*args , **kwargs)
        print("After running function")
        return result
    return wrapper                     


@ decorator_args                             # shortcut Function Decoration
def show_message(message):
    print(message)


# Decorator with return value:

def double_result(Entries):
    def wrapper(*args, **kwargs):
        return Entries(*args, **kwargs) *2
    return wrapper

@ double_result
def add(a, b):
    return a + b


# Nested decorators : (you can use several decorators in one Function)

def star(func):
    def wrapper(*args, **kwargs):
        print("**")
        func(*args, **kwargs)
        print("**")

def exclamation(func):
    def wrapper(*args, **kwargs):
        print("!!")
        func(*args, **kwargs)
        print("!!")
    return wrapper

@ star
@ exclamation
def decorated_name():
    print("My name is Zahra")

## Parameterized Decorator
def repeat(n: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def hello():
    print("hi")

# functools.wraps usage
from functools import wraps

def wraps_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("before run")
        return func(*args, **kwargs)
    return wrapper

@wraps_decorator
def say_hello(name: str) -> str:
    """Returns a greeting."""
    return f"Hello {name}"

print(say_hello("Zahra"))       # Hello Zahra (Before run)
print(say_hello.__name__)       # say_hello (Name of Function)
print(say_hello.__doc__)        # Returns a greeting (The Docstring)



# 7. ---------- Lambda Functions --------

## Simple generator that yields numbers from 0 to 4
def generate_sequence():
    for value in range(5):
        yield value

for seq_item in generate_sequence():
    print("Generated item:", seq_item)


# Generator that yields squares of numbers up to n-1
def generate_squares(limit):
    for idx in range(limit):
        yield idx ** 2

for squared_value in generate_squares(4):
    print("Generated square:", squared_value)


# Batch Processing with Generators: 

def generate_batches(items, batch_size):
    """
    Yields items in batches of given batch_size.
    """
    current_batch = []
    for element in items:
        current_batch.append(element)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch

dataset = range(10)

for batch in generate_batches(dataset, 3):
    print("Processing batch:", batch)

# Generator Expressions (Short Version):
# Simple generator expression
simple_gen_expr = (i for i in range(5))
for _ in range(5):
    print("Generated from expression:", next(simple_gen_expr))

# Conditional generator expression
even_numbers_gen = (x for x in range(10) if x % 2 == 0)
for even_value in even_numbers_gen:
    print("Conditional generator:", even_value)

# Generator with condition:
Even_flow = (digit for digit in range(10) if digit % 2 == 0)
for element in Even_flow:
    print("condotional Generator:", element)                   # Conditional Generator




# Type Hints 
def add_typed(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

def greet_typed_user(user_name: str) -> str:
    """Return a greeting string for the given user."""
    return f"Hi {user_name}"

print(greet_typed_user("Zahra"))



# ========================================================================================================
# STRINGS 
# ========================================================================================================

# Raw string
raw_string = r"Hello\nworld"
print("Raw string:", raw_string)

# Indexing and slicing
sample_text = "Python"
print(sample_text[0])       # 'P'
print(sample_text[-1])      # 'n'
print(sample_text[1:4])     # 'yth'
print(sample_text[::-1])    # 'nohtyp'

# String methods
padded_text = "hello world   "
print("Lower method:", padded_text.lower())              
print("Upper method:", padded_text.upper())              
print("Strip method:", padded_text.strip())              
print("Replace method:", padded_text.replace("world", "Zahra"))  
print("Length:", len(padded_text))                       
print("Slicing:", padded_text[2:7])                      
print("Capitalized:", padded_text.capitalize())          
print("Title case:", padded_text.title())                
print("Swap case:", padded_text.swapcase())              

# Classic string formatting
name = "Zahra"
age = 25
formatted_text = "My name is {n} and I am {a} years old.".format(n=name, a=age)
print(formatted_text)

pi = 3.14159265
print("Pi is approximately {:.2f}".format(pi))  # Two decimal places

# f-strings
print(f"My name is {name} and I am {age} years old.")                       
print(f"In 5 years, I'll be {age + 5}")                                        
print(f"My name in uppercase is {name.upper()}.")                              
print(f"You are {'of legal age' if age >= 18 else 'underage'}.")            

user = {"name": "Zahra", "age": 25}
print(f"{user['name']} is {user['age']} years old.")

# ============================================================================================================
#  LISTS 
# ============================================================================================================

# 1. Types of lists
empty_list = []
numbers_list = [1, 2, 3, 4, 5, 6]
fruits_list = ["apple", "banana", "cherry", "date"]
mixed_list = [1, "apple", True, 3.14]     # Mixed types

# 2. Indexing and slicing
sample_fruit = fruits_list[0]             # 'apple'
last_fruit = fruits_list[-1]              # 'date'
sub_fruits = fruits_list[1:3]             # ['banana', 'cherry']
reversed_fruits = fruits_list[::-1]       # ['date', 'cherry', 'banana', 'apple']

# 3. Mutability and common methods
fruits_list[1] = "Blueberry"              # Change element
fruits_list.append("Kiwi")                # Add to end
fruits_list.insert(1, "Grape")            # Insert at index
fruits_list.extend(["Mango", "Peach"])    # Add multiple
fruits_list.remove("cherry")              # Remove by value
del fruits_list[0]                        # Delete by index
last_item = fruits_list.pop()             # Remove and return last item

# 4. Sorting and reversing
fruits_list.sort()                        # Alphabetical
fruits_list.sort(reverse=True)            # Reverse alphabetical
fruits_list.reverse()                     # Reverse order

# 5. Iteration
for fruit in fruits_list:
    pass  # Loop through elements

for idx, fruit in enumerate(fruits_list):
    pass  # Loop with index

for num, char in zip([1,2,3], ['x','y','z']):
    pass  # Loop multiple iterables

# 6. Membership
exists = "apple" in fruits_list

# 7. Copying lists
import copy
original_list = [3,4]
reference_copy = original_list            # "=" reference
shallow_copy = original_list.copy()       # Shallow copy
deep_copy = copy.deepcopy(original_list)  # Deep copy

# 8. Other useful methods
first_index = fruits_list.index("Blueberry")
count_apple = fruits_list.count("apple")
copied_list = fruits_list.copy()
fruits_list.clear()                      # Empty the list

# --------- LIST COMPREHENSIONS -----------

numbers = [1, 2, 3, 4, 5]

# Basic comprehension
squared = [n*2 for n in numbers]

# With strings
words = ["apple", "banana", "cherry"]
upper_words = [w.upper() for w in words if len(w) > 5]

# Conditional expressions
even_doubled = [n*2 for n in numbers if n % 2 == 0]
labels = ['even' if n % 2 == 0 else 'odd' for n in numbers]
filtered = [n for n in numbers if n > 2 and n % 2 == 0]

# Nested comprehension
matrix = [[1,2,3],[4,5,6],[7,8,9]]
flat = [n for row in matrix for n in row]
even_squared = [n**2 for row in matrix for n in row if n % 2 == 0]
processed = [n**2 if n % 2 == 0 else n**3 for row in matrix for n in row]

import math
sqrt_even = [math.sqrt(n) for row in matrix for n in row if n % 2 == 0]


# =============================================================================================
#  TUPLES 
# =============================================================================================

# 1. Creating tuples
empty_tuple = ()
number_tuple = (1, 2, 3, 4)
mixed_tuple = ("apple", True, 3.14)
tuple_without_parentheses = 5, 6, 7

# 2. Indexing and slicing
sample_tuple = ("Apple", "Banana", "Cherry", "Date")
first_element = sample_tuple[0]         # 'Apple'
last_element = sample_tuple[-1]         # 'Date'
sub_tuple = sample_tuple[1:3]           # ('Banana', 'Cherry')
reversed_tuple = sample_tuple[::-1]     # ('Date', 'Cherry', 'Banana', 'Apple')

# 3. Common tuple operations
numeric_tuple = (1, 2, 3, 4)
tuple_length = len(numeric_tuple)       # Length
count_of_value = numeric_tuple.count(2) # Count occurrences
index_of_value = numeric_tuple.index(3) # Index of first occurrence

# 4. Tuple unpacking
unpack_tuple = (10, 20, 30)
first_value, second_value, third_value = unpack_tuple

# Extended unpacking
first_item, *middle_items, last_item = (1, 2, 3, 4, 5)

# 5. Nested tuples
nested_tuple = ((1, 2), (3, 4), (5, 6))
nested_element = nested_tuple[1][0]   # Access element 3


# ============================================================================================
# Sets
# ============================================================================================

# 1. Creating sets
empty_set = set()
number_set = {1, 2, 3, 4}
fruit_set = {"apple", "banana", "cherry"}

"""
Sets are unordered collections of unique elements.
They are mutable: you can add or remove elements.
"""

# 2. Basic set operations
example_set = {1, 2, 3}

# Add element
example_set.add(4)

# Remove elements
example_set.remove(2)   # Raises error if element not present
example_set.discard(5)  # Does not raise error if element not present

# Pop removes and returns a random element
popped_element = example_set.pop()

# Clear the set
example_set.clear()

# 3. Mathematical operations on sets
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}

union_set = set_a | set_b                  # Union
intersection_set = set_a & set_b           # Intersection
difference_set = set_a - set_b             # Difference
symmetric_difference_set = set_a ^ set_b   # Symmetric Difference

# 4. Conversion between list, tuple, and set
original_list = [1, 2, 2, 3]
converted_set = set(original_list)         # {1, 2, 3}
converted_tuple = tuple(converted_set)     # (1, 2, 3)
converted_list = list(converted_set)       # [1, 2, 3]

# 5. Set comprehensions
numbers_list = [1, 2, 2, 3, 4]
unique_numbers_set = {num for num in numbers_list}

even_numbers_set = {num for num in range(10) if num % 2 == 0}


# =============================================================================================
# Dictionaries
# =============================================================================================

# 1. Creating dictionaries
empty_dict_literal = {}
empty_dict_constructor = dict()

person_dict_literal = {"name": "Alice", "age": 30, "city": "Paris"}
person_dict_constructor = dict(name="Bob", age=30, city="London")

# 2. Accessing values
name_value = person_dict_literal["name"]                 # Access with key
age_value = person_dict_literal.get("age")               # Access with get()
salary_value = person_dict_literal.get("salary", 0)      # Access with default

# 3. Adding and updating values
person_dict_literal["age"] = 26                          # Update existing key
person_dict_literal["salary"] = 5000                     # Add new key

# 4. Deleting values
del person_dict_literal["salary"]                        # Delete key-value
popped_age = person_dict_literal.pop("age")              # Remove key and return value
person_dict_literal.popitem()                            # Remove last inserted key-value
person_dict_literal.clear()                              # Empty the dictionary

# 5. Iterating over dictionary
student_info = {"name": "Zahra", "age": "16"}

for key in student_info:
    value = student_info[key]

for key, value in student_info.items():
    pass  # Standard way to iterate over key-value pairs

dict_keys = student_info.keys()
dict_values = student_info.values()
dict_items = student_info.items()

# 6. Dictionary comprehensions
squares_dict = {num: num**2 for num in range(5)}
even_squares_dict = {num: num**2 for num in range(10) if num % 2 == 0}

# 7. Nested dictionaries
nested_student_dict = {
    "Alice": {"age": 25, "grade": "A"},
    "Bob": {"age": 22, "grade": "B"}
}

# 8. Other useful methods
update_dict = {"name": "Alice"}
update_dict.update({"age": 25, "city": "Paris"})
update_dict.setdefault("country", "France")
copied_dict = update_dict.copy()



# =======================================================================================================
# Modules & Imports
# =======================================================================================================

# 1. Installing and managing libraries with pip
"""
pip install <package_name>       # Install a library
pip uninstall <package_name>     # Uninstall a library
pip list                         # Show installed libraries
pip show <package_name>          # Show details (version, location)
pip install -U <package_name>    # Update to the latest version
"""

# 2. Importing modules
import math_module       

# Using specific functions from a module
from math import sqrt, pow

# Import everything from module (not recommended)
from math import *

# 3. Using a custom module
import math_module as custom_math_module

# Example function usage in custom module
# custom_math_module.greet_user("Zahra")
# sum_result = custom_math_module.sum_up(2, 3, 4, 5, 6)


# ==============================================================================
# File Handling
# ==============================================================================

"""
File handling includes opening, reading, writing, appending, and closing files (txt, csv, json, binary) in Python.

Mode reference:
'r'   - read
'w'   - write (overwrite)
'a'   - append
'x'   - create
'r+'  - read + write
"""

# 1. Opening files
example_file_path = r"path_to_your_file.txt"

# Basic open
file_object = open(example_file_path, "r")

# Using 'with' ensures automatic closing
with open(example_file_path, "r") as file_with_context:
    pass  # File operations here

# 2. Reading files
with open(example_file_path, "r") as read_file:
    full_content = read_file.read()             # Read entire content

with open(example_file_path, "r") as read_file_lines:
    first_line = read_file_lines.readline()    # Read first line
    second_line = read_file_lines.readline()   # Read second line
    all_lines_list = read_file_lines.readlines()  # Read remaining lines into a list

# Iterating over file line by line
with open(example_file_path, "r") as loop_file:
    for line in loop_file:
        stripped_line = line.strip()

# Reading with condition
with open(example_file_path, "r") as conditional_file:
    for line in conditional_file:
        if "python" in line:
            filtered_line = line.strip()

# Reading partial content
with open(example_file_path, "r") as partial_file:
    first_five_chars = partial_file.read(5)
    next_seven_chars = partial_file.read(7)
    remaining_content = partial_file.read()

# Reading binary file
binary_file_path = r"path_to_your_image.png"
with open(binary_file_path, "rb") as binary_file:
    binary_data = binary_file.read()  # Reads all bytes

# tell() and seek()
with open(example_file_path, "r") as seek_file:
    seek_file.read(5)
    current_cursor_position = seek_file.tell()

with open(example_file_path, "r") as move_cursor_file:
    move_cursor_file.seek(0)          # Move cursor to start
    first_six_chars = move_cursor_file.read(6)

    move_cursor_file.seek(7, 0)       # Move cursor with offset
    next_five_chars_after_offset = move_cursor_file.read(5)

"""
seek(offset, whence)
offset = number of bytes (or positions) to move the cursor
whence = reference point (0=start, 1=current, 2=end)
"""

# 3. Writing files
with open(example_file_path, "w") as write_file:
    write_file.write("Name, Age, Score\n")
    write_file.write("Alice, 25, 90\n")
    write_file.write("Bob, 30, 85\n")  # 'w' overwrites the file

with open(example_file_path, "a") as append_file:
    append_file.write("Hello, Zahra\n")
    append_file.write("This is a test file.\n")

# Writing multiple lines from a list
lines_to_append = ["zahra, HTML, 30%\n", "zahra, CSS, 10%\n", "zahra, Python, 80%\n"]
with open(example_file_path, "a") as append_file:
    for line in lines_to_append:
        append_file.write(line)

# ==================================================================================================
# Classes & Object-Oriented Programming
# ===================================================================================================

# 1. ------- Basic class definition ----------
class EmptyClass:
    pass

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 2. ---------- Object instantiation ---------
user_person = Person("Zahra", 22)

# 3. --------- Methods in class -------------
class PersonWithMethods:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, I'm {self.name} and I'm {self.age} years old."

user_with_method = PersonWithMethods("Zahra", 22)

# 4. ------------ Attributes types ---------
class UserAttributes:
    def __init__(self, name, age):
        self.name = name             # Public
        self._name_protected = name  # Protected
        self.__name_private = name   # Private

# Accessing private attribute using name mangling
user_attr = UserAttributes("Zahra")
private_name_access = user_attr._UserAttributes__name_private

# 5. ------------------- Encapsulation example -------------------
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance    # Private attribute

    def deposit(self, amount):
        self.__balance += amount

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount

    def get_balance(self):
        return self.__balance

# 6.-------------- Property decorator example ----------------
class CircleShape:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be non-negative.")
        self._radius = value

# 7.----------------- Inheritance ---------------------
class Vehicle:
    def __init__(self, make, model):
        self.make = make
        self.model = model

    def info(self):
        return f"{self.make} {self.model}"

class Car(Vehicle):
    def __init__(self, make, model, seats):
        super().__init__(make, model)
        self.seats = seats

# 8.---------------- Polymorphism ------------------------
class Shape:
    def area(self):
        raise NotImplementedError

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        import math
        return math.pi * (self.radius ** 2)

# 9.----------- Abstraction (Abstract Base Class) ----------------
from abc import ABC, abstractmethod

class AbstractVehicle(ABC):
    @abstractmethod
    def start_engine(self):
        pass

class Bike(AbstractVehicle):
    def start_engine(self):
        return "Bike engine started"

# 10.----------------- Context manager using 'with' ------------------
class ManagedFile:
    def __init__(self, filepath, mode):
        self.filepath = filepath
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filepath, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

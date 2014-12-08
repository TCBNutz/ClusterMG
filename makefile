py_target=test

all: py clean

py:
	python $(py_target).py

clean:
	@rm -f *.pyc

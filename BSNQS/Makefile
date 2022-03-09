.PHONY: clean
.DEFAULT_GOAL: main.exe

CC = g++
CCMPI = mpicxx
CFLAGS := -Wall -O3 -pedantic -std=c++11
OBJECTS = main.o
VPATH = src
CLASSES = ansatz.cpp model.cpp sampler.cpp read_options.cpp random.cpp
HEADERS = library.h

main.exe : $(OBJECTS)
	$(CC) $(OBJECTS) -o main.exe $(CFLAGS) -larmadillo -framework Accelerate
main.o : main.cpp $(CLASSES) $(HEADERS)
	$(CC) -c main.cpp -o main.o $(CFLAGS)
clean :
	rm *.o *.dat *.wf main.exe
